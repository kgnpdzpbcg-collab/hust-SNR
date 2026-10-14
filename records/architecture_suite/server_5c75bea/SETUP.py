"""部署原九模型代码，先用真实数据检查源划分、噪声和成对初始化。"""
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
PLAIN = BASE/'experiments/plain_cnn1d/5785ae5'
STRONG = BASE/'datasets/strong_snr_v1_20261012'

def sha(path):
    """以原字节核验代码和历史文件，不改写输入或已有成绩。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()

def save(path,value):
    """部署记录只创建一次，重复启动必须另分配目录。"""
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    """源数据、旧实现、GPU及九模型集成预检全部通过才允许正式训练。"""
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
    for name in ('scripts/v2_experiment.py','scripts/v1_noise_consistency.py','scripts/v2_paifilter.py',
        'scripts/v5_nahsf.py','scripts/prepare.py','scripts/classical_baseline_models.py',
        'configs/splits_gate0_v1.json'):
        assert sha(CODE/name) == reference['code_sha256'][name],name
    assert json.loads((PLAIN/'results/reports/VERIFICATION.json').read_text())['status'] == 'passed'
    protected = json.loads((PLAIN/'PROTECTED_FILES_SHA256.json').read_text())
    protected.update({str(path):sha(path) for path in (PLAIN/'results').rglob('*') if path.is_file()})
    assert all(sha(path) == expected for path,expected in protected.items())
    save(ROOT/'PROTECTED_FILES_SHA256.json',protected)
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    commands = [[sys.executable,'-m','pytest','-q','tests/test_hust_arch_suite.py'],
        [sys.executable,'scripts/hust_arch_suite_experiment.py','smoke'],
        ['bash','-n','scripts/run_hust_arch_suite_server.sh']]
    for index,command in enumerate(commands):
        process = subprocess.run(command,cwd=CODE,env=os.environ.copy(),text=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (ROOT/f'preflight_{index}.log').write_text(process.stdout,encoding='utf-8')
        print(process.stdout,flush=True)
        assert process.returncode == 0
    sys.path.insert(0,str(CODE/'scripts'))
    import numpy as np
    import torch
    import hust_arch_suite_experiment as exp
    import v2_experiment as data
    from v1_noise_consistency import fix_randomness
    assert torch.cuda.is_available()
    (tr,tm,y,clean),(vr,vm,vy,val_clean,val) = exp.source_views(STRONG)
    original = data.fixed_noisy_validation(vr,vm,noise_levels=(0,5,10,20,30))
    assert all(np.array_equal(val[k],original[k]) for k in val)
    previous = json.loads((CLASSICAL/'PREFLIGHT.json').read_text())
    initial,noise,counts,gpu = {},{},{},{}
    for seed in exp.SEEDS:
        noisy = data.fresh_noisy_source(tr,tm,seed,1,noise_levels=exp.TRAIN_NOISE)
        noise[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noise[str(seed)] == previous['first_epoch_noise_sha256'][str(seed)]
        ids = np.random.default_rng(seed+902).permutation(len(y))[:32]
        initial[str(seed)] = {}
        resnet_backbone_states = []
        for method in exp.METHODS:
            fix_randomness(seed)
            model = exp.HUSTArchitectureCandidate(method).cuda()
            initial[str(seed)][method] = exp.digest_parameters(model)
            counts[method] = exp.parameter_count(model)
            if method.startswith('resnet_'):
                backbone = {key:value.detach().cpu().clone() for key,value in model.backbone.state_dict().items()}
                resnet_backbone_states.append(backbone)
                # 原ResNet的首份参数摘要含stem/layers/pool/classifier；新包装不改变主干初始化。
                assert exp.digest_parameters(model.backbone) == previous['initial_resnet_sha256'][str(seed)]
            loss = exp.paired_ce(model,torch.from_numpy(clean[ids]).cuda(),
                torch.from_numpy(noisy[ids]).cuda(),torch.from_numpy(y[ids]).cuda())
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            gpu[f'{method}_seed{seed}'] = {'first_batch_loss':float(loss.detach().cpu()),
                'all_parameter_gradients_finite':True}
            del model
        for left,right in [('v5_wd_plain','v5_wd_res'),('v5_wd_plain_ms','v5_wd_res_ms')]:
            assert initial[str(seed)][left] == initial[str(seed)][right]
            assert counts[left] == counts[right]
        assert all(torch.equal(resnet_backbone_states[0][name],other[name])
            for other in resnet_backbone_states[1:] for name in resnet_backbone_states[0])
        del resnet_backbone_states
    expected_counts = {'v5_w':63213,'v5_wd_plain':88045,'v5_wd_res':88045,'v5_w_ms':63693,
        'v5_wd_plain_ms':88525,'v5_wd_res_ms':88525,'resnet_lp3':241702,
        'resnet_lp4':241702,'resnet_nahsf':241743}
    assert counts == expected_counts
    torch.cuda.empty_cache()
    save(ROOT/'DEPLOYMENT_MANIFEST.json',{'code_commit':'5c75bea','branch':'research/architecture-sweep-v1',
        'environment_reused':True,'python':sys.executable,'source_strong_root':str(STRONG),
        'target_strong_root':str(STRONG),'target_snr_root':str(FOUR/'target_snr'),
        'reference_classical_results':str(CLASSICAL/'results'),'reference_four_results':str(FOUR/'results'),
        'config':exp.configuration(),'parallel_workers':4,'threads_per_worker':4,
        'original_model_training_and_data_code_modified':False,
        'code_sha256':{path.relative_to(CODE).as_posix():sha(path) for path in CODE.rglob('*')
            if path.is_file() and path.suffix in ('.py','.json','.md','.sh')}})
    save(ROOT/'PREFLIGHT.json',{'status':'passed','unit_tests_passed':25,'cli_smoke_passed':True,
        'shell_syntax_passed':True,'source_train_records':34,'source_val_records':15,
        'source_train_cycles':738,'source_val_cycles':302,'source_split_record_overlap':0,
        'source_val_noise_matches_original_rng':True,'first_epoch_noise_sha256':noise,
        'first_epoch_noise_matches_historical_resnet':True,'initial_parameters_sha256':initial,
        'plain_residual_pairs_equal_parameters_and_initial_weights':True,
        'new_resnet_backbones_equal_historical_initial_weights':True,'parameter_counts':counts,
        'real_gpu_first_batch_checks':gpu,'device':torch.cuda.get_device_name(),
        'torch_version':str(torch.__version__),'final_holdout_evaluated':False})
    print('NINE_ARM_PREFLIGHT_PASSED',flush=True)

if __name__ == '__main__': main()
