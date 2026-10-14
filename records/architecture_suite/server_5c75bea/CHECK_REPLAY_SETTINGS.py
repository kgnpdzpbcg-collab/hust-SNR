"""诊断核验脚本与原正式推理的CUDA设置差异，权重/数据保持只读。"""
from pathlib import Path
import json
import os
import sys

os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
sys.path.insert(0,str(ROOT/'code/scripts'))
import numpy as np
import torch
import hust_arch_suite_experiment as exp
from v1_noise_consistency import fix_randomness

def flags():
    """记录会影响FP32卷积数值和算法选择的真实开关。"""
    return {'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}

deploy = json.loads((ROOT/'DEPLOYMENT_MANIFEST.json').read_text())
raw,metadata = exp.target_raw(10,snr_root=Path(deploy['target_snr_root']),
    strong_root=Path(deploy['target_strong_root']),dev_export_root=None)
x = exp.data.raw_cycles(raw,metadata)
truth = [row['label'] for row in metadata]
default = flags()
models,expected,comparisons = {},{},[]
for method in exp.METHODS:
    for seed in exp.SEEDS:
        folder = OUT/f'models/{method}/seed{seed}'
        checkpoint = torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        model = exp.HUSTArchitectureCandidate(method).cuda().eval()
        model.load_state_dict(checkpoint['state'],strict=True)
        original = json.loads((OUT/f'evaluations/{method}/seed{seed}/target_dev_snr10.json').read_text())
        assert original['truth'] == truth
        predicted = exp.infer(model,x,torch.device('cuda'))
        mismatch = np.flatnonzero(predicted != original['predicted'])
        models[method,seed],expected[method,seed] = model,np.array(original['predicted'])
        comparisons.append({'method':method,'seed':seed,'default_mismatched_cycles':len(mismatch),
            'default_mismatch_indices':mismatch.tolist()})
for row in comparisons:
    method,seed = row['method'],row['seed']
    # 使用正式evaluate相同的原fix_randomness，不更改网络结构、权重或输入。
    fix_randomness(seed)
    predicted = exp.infer(models[method,seed],x,torch.device('cuda'))
    row['original_settings_mismatched_cycles'] = int(np.count_nonzero(predicted != expected[method,seed]))
    assert row['original_settings_mismatched_cycles'] == 0
record = {'default_verifier_settings':default,'original_evaluation_settings':flags(),
    'condition':'snr10','models_checked':27,'comparisons':comparisons,
    'original_settings_all_predictions_exact':True,'model_or_data_changes':False}
path = ROOT/'REPLAY_SETTINGS_DIAGNOSIS.json'
assert not path.exists()
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('DEFAULT_SETTINGS',json.dumps(default))
print('ORIGINAL_SETTINGS',json.dumps(record['original_evaluation_settings']))
print('DEFAULT_MISMATCHES',json.dumps([row for row in comparisons if row['default_mismatched_cycles']]))
print('MATCHED_ORIGINAL_SETTINGS_ALL_27_EXACT')
