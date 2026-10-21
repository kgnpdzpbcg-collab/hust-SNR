"""部署本地V6快照，核对验证SNR修复、源输入和记录级健康参考段。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile

RUN=Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v6/local_4b59ac8c_20261021')
V5=RUN.parents[1]/'jung_hang_cycle_v5/164f1c7_20261021'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8',newline='\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """保留原本地源码与唯一修复补丁，环境及冻结数据读取器保持原版本。"""
    assert not RUN.exists(),'已有V6目录，拒绝覆盖'
    RUN.mkdir(parents=True);(RUN/'logs').mkdir()
    with tarfile.open('/tmp/hust_jung_hang_v6_local_4b59ac8c.tar') as archive:
        archive.extractall(RUN,filter='data')
    source=read(RUN/'DEPLOYMENT_SOURCE_MANIFEST.json')
    for name,h in source['deployed_code_sha256'].items():assert sha(RUN/'code'/name)==h,name
    previous=read(V5/'DEPLOYMENT_MANIFEST.json')
    protected=read(V5/'PROTECTED_FILES_BEFORE.json')
    for name,h in protected.items():assert sha(Path(name))==h,name
    for path in V5.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts:
            protected[str(path)]=sha(path)
    write(RUN/'PROTECTED_FILES_BEFORE.json',protected)
    references={k:previous[k] for k in ('reference_root','source_root','additional_root','previous_output','reference_reader_commit')}
    write(RUN/'REFERENCE_PATHS.json',references)
    # 先导入参考数据模块，再导入V6模型，确保prepare的数据根目录指向原冻结快照。
    bootstrap='''"""只适配服务器数据路径的启动包装器，不改V6算法或读取器。"""
from pathlib import Path
import json,runpy,sys
root=Path(__file__).resolve().parent
paths=json.loads((root/'REFERENCE_PATHS.json').read_text())
sys.path.insert(0,str(Path(paths['reference_root'])/'code/scripts'))
import source_cycle_benchmark,source_cycle_split
sys.path.insert(0,str(root/'code/scripts'))
entry=root/'code/scripts/jung_hang_cycle_v6.py'
sys.argv=[str(entry),*sys.argv[1:]]
runpy.run_path(str(entry),run_name='__main__')
'''
    (RUN/'ENTRY_BOOTSTRAP.py').write_text(bootstrap,encoding='utf8',newline='\n')
    os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                      PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    scripts=RUN/'code/scripts'
    assert sha(scripts/'prepare.py')==sha(Path(references['reference_root'])/'code/scripts/prepare.py')
    subprocess.run([sys.executable,'-m','py_compile',*[str(scripts/name) for name in
        ('jung_hang_models_v6.py','compare_jung_hang_v6.py','jung_hang_optim_v6.py','jung_hang_cycle_v6.py')]],check=True)
    with (RUN/'logs/pytest.log').open('w') as log:
        p=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_jung_hang_models_v3.py',
            'tests/test_jung_hang_models_v6.py','tests/test_jung_hang_cycle_v6_noise.py'],cwd=RUN/'code',stdout=log,stderr=subprocess.STDOUT)
    print((RUN/'logs/pytest.log').read_text(),flush=True);assert p.returncode==0,'正式pytest失败'
    args=argparse.Namespace(config=RUN/'code/configs/jung_hang_v6_v1.json',reference_root=Path(references['reference_root']),
         source_root=Path(references['source_root']),additional_root=Path(references['additional_root']),
         previous_output=Path(references['previous_output']),output=RUN/'results')
    cmd=[sys.executable,'-u',str(RUN/'ENTRY_BOOTSTRAP.py'),'preflight','--config',str(args.config)]
    for key in ('reference_root','source_root','additional_root','previous_output'):
        cmd+=['--'+key.replace('_','-'),references[key]]
    cmd+=['--output',str(args.output)]
    with (RUN/'logs/preflight.log').open('w') as log:
        subprocess.run(cmd,cwd=RUN/'code',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    print((RUN/'logs/preflight.log').read_text(),flush=True)
    sys.path.insert(0,str(args.reference_root/'code/scripts'))
    import source_cycle_benchmark as ref
    import source_cycle_split as split
    sys.path.insert(0,str(scripts))
    import numpy as np
    import torch
    import jung_hang_cycle_v6 as runner
    torch.set_num_threads(2)
    checks=[]
    for method in ('jung','hang'):
        c,_,paper,prov,raw,meta,y,clean,vy,val=runner.source_data(args,method)
        indices=[int(np.flatnonzero(y==label)[0]) for label in range(6)]
        ref.fix_randomness(42);model=paper.get_model(method,c).cuda().train()
        loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(clean[indices]).cuda()),torch.tensor(y[indices],device='cuda'))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        checks.append({'method':method,'train_shape':list(clean.shape),'val_shape':list(val['clean'].shape),
                       'parameters':sum(p.numel() for p in model.parameters()),'finite_cuda_gradients':True})
        del model
    # 实际源记录上核对修复后的验证分支；该加噪函数允许有限长度高斯实现的功率波动。
    rid=sorted(raw)[0];t,x,w=raw[rid];snr_checks=[]
    for method in ('jung','hang'):
        for snr in c['val_noise_db']:
            noisy=runner.noise_records({rid:(t,x,w)},42,1,method=method,meta=[],config=c,snr=snr)[rid][1]
            actual=10*np.log10(np.mean(x*x,axis=0)/np.mean((noisy-x)**2,axis=0))
            assert np.max(np.abs(actual-snr))<.5
            snr_checks.append({'method':method,'requested_snr_db':snr,'actual_phase_snr_db':actual.tolist()})
    (raw0,train_meta,_,_),(_,val_meta,_,_,_)=split.source_views(args.source_root)
    lo,hi=c['reference_window_s']
    overlap=[r for r in val_meta if r['start']<hi and r['end']>lo]
    healthy_overlap=[r for r in overlap if int(r['label'])==0]
    reference_audit={'reference_window_s':[lo,hi],'source_val_windows_overlapping_reference':len(overlap),
        'source_val_healthy_windows_overlapping_reference':len(healthy_overlap),'source_val_total_windows':len(val_meta),
        'source_training_reference_estimates_use_full_record_healthy_segment':True,
        'target_reference_estimates_use_same_condition_noisy_record':True,
        'clean_target_counterpart_used':False,'reference_is_not_restricted_to_current_window':True,
        'note':'record-level reference calibration is part of V6; reference can overlap held-out source-Val times and can include later samples relative to healthy test windows; this is not a strictly window-only preprocessing protocol.'}
    write(RUN/'REFERENCE_PROTOCOL_AUDIT.json',reference_audit)
    phase=read(V5/'SOURCE_PHASE_ORDER_AUDIT.json')
    assert c['hang_phase_angles_deg']==previous['config']['hang_phase_angles_deg']
    write(RUN/'SOURCE_PHASE_ORDER_AUDIT.json',phase)
    manifest={**references,'local_source_snapshot_sha256':source['local_snapshot_sha256'],'local_run_name':source['run_name'],
        'model_version':'local V6 plus validation SNR branch fix','base_commit':source['base_commit'],'previous_v5_root':str(V5),
        'config':c,'config_sha256':sha(args.config),'source_split':prov,'interpreter':sys.executable,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'tests_passed':26,'cuda_gradient_checks':checks,'real_source_validation_snr_checks':snr_checks,
        'reference_protocol_audit':reference_audit,'local_original_source_preserved':True,'validation_fix_patch':str(RUN/'VALIDATION_SNR_FIX.patch'),
        'code_sha256':source['deployed_code_sha256'],'protected_files':len(protected),'workers':4,'cpu_threads_per_worker':2,'final_holdout_evaluated':False}
    write(RUN/'DEPLOYMENT_MANIFEST.json',manifest)
    print(json.dumps({'run':str(RUN),'checks':checks,'reference_protocol_audit':reference_audit,'protected_files':len(protected)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
