"""经典基线部署及真实数据预检：新目录、原代码、旧环境和冻结文件。"""
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
XU = BASE/'experiments/xu2025_full/bb2f219'
STRONG = BASE/'datasets/strong_snr_v1_20261012'

def sha(path):
    """校验实际源代码、模型及数据字节，不修改现有文件。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()

def save(path, value):
    """每个部署记录只创建一次，拒绝覆盖。"""
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    """先检查模型/数据/噪声和GPU，再允许正式训练。"""
    assert not CODE.exists() and not (ROOT/'results').exists()
    CODE.mkdir()
    with tarfile.open(ROOT/'code.tar') as archive: archive.extractall(CODE, filter='data')
    (CODE/'data/target_dev').mkdir(parents=True)
    (CODE/'outputs').mkdir()
    for link, target in ((CODE/'data/source', FOUR/'code/data/source'),
        (CODE/'data/target_dev/clean', FOUR/'code/data/target_dev/clean'),
        (CODE/'outputs/gate0_v1', FOUR/'code/outputs/gate0_v1')):
        link.symlink_to(target, target_is_directory=True)
    reference = json.loads((XU/'DEPLOYMENT_MANIFEST.json').read_text())
    for name in ('scripts/v2_experiment.py', 'scripts/v1_noise_consistency.py',
        'scripts/v2_paifilter.py', 'scripts/prepare.py', 'configs/splits_gate0_v1.json'):
        assert sha(CODE/name) == reference['code_sha256'][name], name
    protected = json.loads((XU/'ORIGINAL_FILES_SHA256.json').read_text())
    # 上一轮完整可视化也是只读历史；包含高维特征但没有最终保留集波形。
    for directory in (XU/'results', BASE/'experiments/six_model_tsne/f398d30/results'):
        protected.update({str(p): sha(p) for p in directory.rglob('*') if p.is_file()})
    assert all(sha(p) == expected for p, expected in protected.items())
    save(ROOT/'PROTECTED_FILES_SHA256.json', protected)
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_classical_baselines.py']
    test = subprocess.run(command, cwd=CODE, env=os.environ.copy(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ROOT/'tests.log').write_text(test.stdout, encoding='utf-8')
    print(test.stdout, flush=True)
    assert test.returncode == 0
    sys.path.insert(0, str(CODE/'scripts'))
    import numpy as np
    import torch
    import classical_baseline_experiment as exp
    import v2_experiment as data
    from v1_noise_consistency import fix_randomness
    tr, tm, y = data.source_data('train')
    vr, vm, _ = data.source_data('val')
    assert (len(tr), len(vr), len(tm), len(vm)) == (34, 15, 738, 302)
    val = exp.noise_source_val(STRONG, vr, vm)
    reference_val = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5, 10, 20, 30))
    assert all(np.array_equal(val[k], reference_val[k]) for k in val)
    clean = data.raw_cycles(tr, tm)
    previous = json.loads((FOUR/'PREFLIGHT.json').read_text())
    noise_sha = {}
    initial = {}
    for seed in (42, 43, 44):
        noisy = data.fresh_noisy_source(tr, tm, seed, 1)
        noise_sha[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noise_sha[str(seed)] == previous['first_epoch_noise_sha256'][str(seed)]
        fix_randomness(seed)
        model = exp.FivePhaseResNet1D().cuda()
        initial[str(seed)] = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
            for p in model.parameters())).hexdigest()
        ids = np.random.default_rng(seed+902).permutation(len(y))[:32]
        loss = exp._resnet_loss(model, torch.from_numpy(clean[ids]).cuda(),
            torch.from_numpy(noisy[ids]).cuda(), torch.from_numpy(y[ids]).cuda())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        count = sum(p.numel() for p in model.parameters())
        del model
    torch.cuda.empty_cache()
    save(ROOT/'DEPLOYMENT_MANIFEST.json', {'code_commit': '4860c7766a58cd1f0ffb9b3d92195edb3a3ecc25',
        'branch': 'research/classical-baselines-v1', 'environment_reused': True,
        'python': sys.executable, 'source_strong_root': str(STRONG), 'target_strong_root': str(STRONG),
        'target_snr_root': str(FOUR/'target_snr'), 'parallel_workers': 4, 'threads_per_worker': 4,
        'code_sha256': {p.relative_to(CODE).as_posix(): sha(p) for p in CODE.rglob('*')
            if p.is_file() and p.suffix in ('.py', '.md', '.json', '.sh')},
        'config': exp.config(), 'original_model_and_data_code_unchanged': True})
    save(ROOT/'PREFLIGHT.json', {'status': 'passed', 'unit_tests': 11,
        'source_train_records': 34, 'source_val_records': 15,
        'source_train_cycles': 738, 'source_val_cycles': 302,
        'first_epoch_noise_sha256': noise_sha, 'initial_resnet_sha256': initial,
        'fixed_source_val_matches_original_rng': True, 'resnet_parameters': count,
        'real_gpu_gradients_finite': True, 'torch': str(torch.__version__),
        'device': torch.cuda.get_device_name(), 'final_holdout_evaluated': False})
    print('CLASSICAL_PREFLIGHT_PASSED', flush=True)

if __name__ == '__main__': main()
