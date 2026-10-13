"""七模型只执行固定权重特征提取和降维，两阶段顺序运行并记录命令。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
deploy = json.loads((ROOT/'DEPLOYMENT_MANIFEST.json').read_text())
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    CUBLAS_WORKSPACE_CONFIG=':4096:8')
assert not (ROOT/'results').exists()
started = time.monotonic()
stages = []
for stage in ('extract','project'):
    command = [sys.executable,'-u','scripts/seven_model_tsne.py',stage,'--output',str(ROOT/'results')]
    if stage == 'extract': command += ['--base',deploy['base'],'--classical-results',deploy['classical_results']]
    begin = time.monotonic()
    with (ROOT/f'{stage}.log').open('x') as log:
        process = subprocess.run(command,cwd=ROOT/'code',env=os.environ.copy(),stdout=log,
            stderr=subprocess.STDOUT,timeout=1800)
    stages.append({'stage':stage,'command':command,'exit_code':process.returncode,
        'elapsed_seconds':time.monotonic()-begin})
    (ROOT/'RUN_STATUS.json').write_text(json.dumps({'status':'failed' if process.returncode else 'running',
        'stages':stages,'elapsed_seconds':time.monotonic()-started},ensure_ascii=False,indent=2)+'\n')
    if process.returncode: raise SystemExit(process.returncode)
(ROOT/'RUN_STATUS.json').write_text(json.dumps({'status':'completed','stages':stages,
    'elapsed_seconds':time.monotonic()-started},ensure_ascii=False,indent=2)+'\n')
print('SEVEN_MODEL_TSNE_RUN_COMPLETED',flush=True)
