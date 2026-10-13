"""部署GitHub原PlainCNN1D代码，复用服务器环境、已冻结数据及噪声。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT/'code'
FOUR = BASE/'experiments/four_models_expanded_val/56513c7'
CLASSICAL = BASE/'experiments/classical_baselines/4860c77'
SEVEN = BASE/'experiments/seven_model_tsne/d8194e8'
STRONG = BASE/'datasets/strong_snr_v1_20261012'

def sha(path):
    """按实际字节保护输入实现、冻结数据、旧权重和已有报告。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()

def save(path,value):
    """本轮部署目录只创建一次，拒绝覆盖其他任务输出。"""
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    """代码与真实源数据检查通过后才能启动三种子GPU训练。"""
    assert not CODE.exists() and not (ROOT/'results').exists()
    CODE.mkdir()
    with tarfile.open(ROOT/'code.tar') as archive: archive.extractall(CODE,filter='data')
    (CODE/'data/target_dev').mkdir(parents=True)
    (CODE/'outputs').mkdir()
    for link,target in ((CODE/'data/source',FOUR/'code/data/source'),
        (CODE/'data/target_dev/clean',FOUR/'code/data/target_dev/clean'),
        (CODE/'outputs/gate0_v1',FOUR/'code/outputs/gate0_v1')):
        link.symlink_to(target,target_is_directory=True)
    reference = json.loads((CLASSICAL/'DEPLOYMENT_MANIFEST.json').read_text())
    for name in ('scripts/classical_baseline_experiment.py','scripts/classical_baseline_models.py',
        'scripts/v2_experiment.py','scripts/v1_noise_consistency.py','scripts/v2_paifilter.py',
        'scripts/prepare.py','configs/splits_gate0_v1.json','configs/classical_baselines_v1.json'):
        assert sha(CODE/name) == reference['code_sha256'][name],name
    assert json.loads((CLASSICAL/'results/reports/VERIFICATION.json').read_text())['status'] == 'passed'
    assert json.loads((SEVEN/'results/VERIFICATION.json').read_text())['status'] == 'passed'
    protected = json.loads((SEVEN/'results/PROTECTED_FILES_SHA256.json').read_text())
    protected.update({str(path):sha(path) for path in (SEVEN/'results').rglob('*') if path.is_file()})
    assert all(sha(path) == expected for path,expected in protected.items())
    save(ROOT/'PROTECTED_FILES_SHA256.json',protected)
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    commands = [[sys.executable,'-m','pytest','-q','tests/test_plain_cnn1d_control.py'],
        [sys.executable,'scripts/plain_cnn1d_experiment.py','smoke'],
        ['bash','-n','scripts/run_plain_cnn1d_server.sh']]
    for index,command in enumerate(commands):
        completed = subprocess.run(command,cwd=CODE,env=os.environ.copy(),text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (ROOT/f'preflight_{index}.log').write_text(completed.stdout,encoding='utf-8')
        print(completed.stdout,flush=True)
        assert completed.returncode == 0
    sys.path.insert(0,str(CODE/'scripts'))
    import numpy as np
    import torch
    import plain_cnn1d_experiment as experiment
    import classical_baseline_experiment as old
    import v2_experiment as data
    from v1_noise_consistency import fix_randomness
    tr,tm,y = data.source_data('train')
    vr,vm,vy = data.source_data('val')
    assert (len(tr),len(vr),len(tm),len(vm)) == (34,15,738,302)
    assert not (set(tr) & set(vr))
    val = old.noise_source_val(STRONG,vr,vm)
    expected = data.fixed_noisy_validation(vr,vm,noise_levels=(0,5,10,20,30))
    assert all(np.array_equal(val[k],expected[k]) for k in val)
    clean = data.raw_cycles(tr,tm)
    old_preflight = json.loads((CLASSICAL/'PREFLIGHT.json').read_text())
    initial,noise,losses = {},{},{}
    for seed in (42,43,44):
        noisy = data.fresh_noisy_source(tr,tm,seed,1)
        noise[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noise[str(seed)] == old_preflight['first_epoch_noise_sha256'][str(seed)]
        fix_randomness(seed)
        model = experiment.FivePhasePlainCNN1D().cuda()
        initial[str(seed)] = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        ids = np.random.default_rng(seed+902).permutation(len(y))[:32]
        loss = old._resnet_loss(model,torch.from_numpy(clean[ids]).cuda(),
            torch.from_numpy(noisy[ids]).cuda(),torch.from_numpy(y[ids]).cuda())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        losses[str(seed)] = float(loss.detach().cpu())
        parameter_count = sum(p.numel() for p in model.parameters())
        assert not any('shortcut' in name for name in model.state_dict())
        del model
    torch.cuda.empty_cache()
    save(ROOT/'DEPLOYMENT_MANIFEST.json',{'code_commit':'5785ae5','branch':'research/plain-cnn1d-control-v1',
        'python':sys.executable,'environment_reused':True,'source_strong_root':str(STRONG),
        'target_strong_root':str(STRONG),'target_snr_root':str(FOUR/'target_snr'),
        'reference_classical_results':str(CLASSICAL/'results'),'parallel_workers':3,'threads_per_worker':4,
        'config':experiment.configuration(),'model_and_training_code_modified':False,
        'code_sha256':{path.relative_to(CODE).as_posix():sha(path) for path in CODE.rglob('*')
            if path.is_file() and path.suffix in ('.py','.json','.sh','.md')}})
    save(ROOT/'PREFLIGHT.json',{'status':'passed','unit_tests_passed':8,'smoke_passed':True,
        'shell_syntax_passed':True,'source_train_records':34,'source_val_records':15,
        'source_train_cycles':738,'source_val_cycles':302,'source_split_record_overlap':0,
        'source_val_noise_matches_original_rng':True,'first_epoch_noise_sha256':noise,
        'first_epoch_noise_matches_previous_resnet':True,'initial_weights_sha256':initial,
        'real_gpu_first_batch_loss':losses,'all_parameter_gradients_finite':True,
        'plain_cnn_parameters':parameter_count,'reference_resnet_parameters':old_preflight['resnet_parameters'],
        'device':torch.cuda.get_device_name(),'torch_version':str(torch.__version__),
        'final_holdout_evaluated':False})
    print('PLAIN_CNN1D_PREFLIGHT_PASSED',parameter_count,flush=True)

if __name__ == '__main__': main()
