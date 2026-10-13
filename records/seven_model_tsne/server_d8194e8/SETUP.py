"""独立部署七模型展示入口，复用冻结输入及模型实现，不访问最终保留集。"""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT/'code'
CLASSICAL = BASE/'experiments/classical_baselines/4860c77'
FOUR = BASE/'experiments/four_models_expanded_val/56513c7'
assert not CODE.exists() and not (ROOT/'results').exists()
assert json.loads((CLASSICAL/'results/reports/VERIFICATION.json').read_text())['status'] == 'passed'
CODE.mkdir()
with tarfile.open(ROOT/'code.tar') as archive: archive.extractall(CODE,filter='data')
(CODE/'data/target_dev').mkdir(parents=True)
(CODE/'outputs').mkdir()
(CODE/'data/target_dev/clean').symlink_to(FOUR/'code/data/target_dev/clean',target_is_directory=True)
(CODE/'outputs/gate0_v1').symlink_to(FOUR/'code/outputs/gate0_v1',target_is_directory=True)
reference = json.loads((CLASSICAL/'DEPLOYMENT_MANIFEST.json').read_text())
protected_code = ['scripts/classical_baseline_models.py','scripts/classical_baseline_experiment.py',
    'scripts/v2_experiment.py','scripts/v2_paifilter.py','scripts/four_models_expanded_val.py',
    'scripts/fixed_half_experiment.py','scripts/xu2025_full_model.py','scripts/xu2025_full_experiment.py']
assert all(hashlib.sha256((CODE/name).read_bytes()).hexdigest() == reference['code_sha256'][name]
    for name in protected_code)
manifest = {'code_commit':'d8194e85efc1ba3c7da384dce2ea336a7a0e7c4b',
    'python':sys.executable,'environment_reused':True,'base':str(BASE),
    'classical_results':str(CLASSICAL/'results'),'original_model_and_input_code_unchanged':True,
    'config':json.loads((CODE/'configs/seven_model_tsne_v1.json').read_text()),
    'code_sha256':{p.relative_to(CODE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in CODE.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md')}}
(ROOT/'DEPLOYMENT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('SEVEN_MODEL_VISUALIZATION_SETUP_PASSED')
