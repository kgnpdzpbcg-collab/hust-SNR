"""V5独立部署与源域预检；不读取目标波形、不修改仓库模型代码。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile

RUN=Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v5/164f1c7_20261021')
V4=RUN.parents[1]/'jung_hang_cycle_v4/a0c78d8_20261019'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf8',newline='\n')


def main():
    """先完成分支要求的THD/相序检查，是否启动训练留在预检结果审阅后决定。"""
    assert not RUN.exists(),'已有V5目录，拒绝覆盖'
    RUN.mkdir(parents=True);(RUN/'logs').mkdir()
    with tarfile.open('/tmp/hust_jung_hang_v5_164f1c7.tar') as archive:
        archive.extractall(RUN/'code',filter='data')
    previous=read(V4/'DEPLOYMENT_MANIFEST.json')
    protected=read(V4/'PROTECTED_FILES_BEFORE.json')
    for name,h in protected.items():
        assert sha(Path(name))==h,name
    for path in V4.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts:
            protected[str(path)]=sha(path)
    write(RUN/'PROTECTED_FILES_BEFORE.json',protected)
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                      PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    scripts=RUN/'code/scripts'
    files=[scripts/name for name in ('jung_hang_models_v5.py','compare_jung_hang_v5.py',
                                    'jung_hang_optim_v5.py','jung_hang_cycle_v5.py','jung_hang_v5_audit.py')]
    subprocess.run([sys.executable,'-m','py_compile',*map(str,files)],check=True)
    tests=['tests/test_jung_hang_models.py','tests/test_jung_hang_models_v3.py',
           'tests/test_jung_hang_models_v4.py','tests/test_jung_hang_cycle_v3_noise.py','tests/test_jung_hang_models_v5.py']
    with (RUN/'logs/pytest.log').open('w') as log:
        p=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=RUN/'code',stdout=log,stderr=subprocess.STDOUT)
    print((RUN/'logs/pytest.log').read_text(),flush=True)
    assert p.returncode==0,'CPU测试失败'
    args=argparse.Namespace(config=RUN/'code/configs/jung_hang_v5_v1.json',
        reference_root=Path(previous['reference_root']),source_root=Path(previous['source_root']),
        additional_root=Path(previous['additional_root']),previous_output=Path(previous['previous_output']),output=RUN/'results')
    cmd=[sys.executable,'-u',str(scripts/'jung_hang_cycle_v5.py'),'preflight','--config',str(args.config),
         '--reference-root',str(args.reference_root),'--source-root',str(args.source_root),
         '--additional-root',str(args.additional_root),'--previous-output',str(args.previous_output),'--output',str(args.output)]
    with (RUN/'logs/preflight.log').open('w') as log:
        subprocess.run(cmd,cwd=RUN/'code',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    print((RUN/'logs/preflight.log').read_text(),flush=True)
    sys.path.insert(0,str(scripts))
    import numpy as np
    import torch
    import jung_hang_cycle_v5 as runner
    torch.set_num_threads(2)
    checks=[]
    for method in ('jung','hang'):
        config,ref,paper,prov,raw,tm,y,clean,vy,val=runner.source_data(args,method)
        ids=[int(np.flatnonzero(y==label)[0]) for label in range(6)]
        ref.fix_randomness(42)
        model=paper.get_model(method,config).cuda().train()
        loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(clean[ids]).cuda()),torch.tensor(y[ids],device='cuda'))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        checks.append({'method':method,'train_shape':list(clean.shape),'val_shape':list(val['clean'].shape),
                       'train_class_counts':np.bincount(y,minlength=6).tolist(),'val_class_counts':np.bincount(vy,minlength=6).tolist(),
                       'parameters':sum(p.numel() for p in model.parameters()),'finite_cuda_gradients':True,
                       'all_zero_training_inputs':int(np.sum(np.all(clean==0,axis=tuple(range(1,clean.ndim))))),
                       'unique_input_hashes_by_class':[len({hashlib.sha256(a.tobytes()).hexdigest() for a in clean[y==label]}) for label in range(6)]})
        del model
    # 以健康段基波拟合核对配置的通道顺序。这里记录数据证据，不伪称硬件接线认证。
    phase_rows=[]
    for rid in sorted(raw):
        t,x,w=raw[rid];mask=(t>=.1)&(t<=.45);freq=float(np.median(w[mask]))
        design=np.column_stack((np.cos(freq*t[mask]),np.sin(freq*t[mask]),np.ones(int(mask.sum()))))
        coef=np.linalg.lstsq(design,x[mask],rcond=None)[0]
        phasor=coef[0]-1j*coef[1]
        relative=np.rad2deg(np.angle(phasor/phasor[0]))
        expected=-np.array(config['hang_phase_angles_deg'],dtype=float)
        error=(relative-expected+180)%360-180
        phase_rows.append({'record_id':str(rid),'relative_phase_deg':relative.tolist(),'absolute_wrapped_error_deg':np.abs(error).tolist()})
    phase_errors=np.array([r['absolute_wrapped_error_deg'] for r in phase_rows])
    phase_summary={'source_healthy_fit_records':len(phase_rows),'expected_phasor_angles_deg':(-np.array(config['hang_phase_angles_deg'])).tolist(),
                   'median_absolute_phase_errors_deg':np.median(phase_errors,axis=0).tolist(),'max_absolute_phase_errors_deg':phase_errors.max(axis=0).tolist(),
                   'note':'source waveform evidence only; does not certify physical hardware wiring','records':phase_rows}
    write(RUN/'SOURCE_PHASE_ORDER_AUDIT.json',phase_summary)
    audit=read(args.output/'PREFLIGHT_PAPER_AUDIT.json')
    manifest={key:previous[key] for key in ('reference_root','source_root','additional_root','previous_output','reference_reader_commit')}
    manifest.update(model_branch='research/jung-hang-paper-fidelity-v5',model_commit='164f1c78778b30da7cc18cb71b38e154f56b750d',
        previous_v4_root=str(V4),config=config,config_sha256=sha(args.config),source_split=prov,interpreter=sys.executable,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),tests_log=str(RUN/'logs/pytest.log'),cuda_gradient_checks=checks,
        source_paper_audit=audit,source_phase_order_audit=str(RUN/'SOURCE_PHASE_ORDER_AUDIT.json'),
        code_sha256={p.relative_to(RUN/'code').as_posix():sha(p) for p in (RUN/'code').rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts},
        final_holdout_evaluated=False,workers=4,cpu_threads_per_worker=2,upstream_code_modified=False,protected_files=len(protected))
    write(RUN/'DEPLOYMENT_MANIFEST.json',manifest)
    print(json.dumps({'run':str(RUN),'checks':checks,'source_paper_audit':audit,'phase_error_max_deg':phase_errors.max(axis=0).tolist(),
                      'protected_files':len(protected)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
