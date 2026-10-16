"""在服务器逐阶段调用固定权重入口，保存真实命令、日志、耗时和退出码。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

RUN=Path('/home/wang/projects/会议论文/hust/experiments/b1_seven_snr_fixed_weights/20261016')
ROOT=RUN.parents[2]
stage=sys.argv[1]
assert stage in ('preflight','evaluate','verify','summarize')
log_path=RUN/'logs'/f'{stage}.log'
assert not log_path.exists(),'拒绝重复执行已经有日志的阶段'
environment=os.environ.copy()
environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
command=[sys.executable,'-u',str(RUN/'code/scripts/b1_seven_snr_fixed_weights.py'),stage,
    '--baseline-root',str(ROOT/'experiments/source_cycle_benchmark/4728e9b'),
    '--ablation-root',str(ROOT/'experiments/b1_paper_ablations/d9947f0_20261016'),
    '--additional-root',str(ROOT/'experiments/additional_snr_seed42_44/20261015_snr4_m4_m8'),
    '--source-root',str(ROOT/'datasets/source_cycle_split_v2'),'--output',str(RUN/'output')]
state_path=RUN/'STAGE_STATUS.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':[]}
started=time.time()
state['current_stage']=stage
state['state']='running'
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with log_path.open('w',encoding='utf8') as log:
    result=subprocess.run(command,cwd=RUN/'code',env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=600)
state['jobs'].append({'stage':stage,'command':command,'started_unix':started,'seconds':time.time()-started,'exit_code':result.returncode})
state['state']='passed' if result.returncode==0 else 'failed'
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(log_path.read_text(encoding='utf8')[-6500:],flush=True)
print(json.dumps(state['jobs'][-1],ensure_ascii=False),flush=True)
raise SystemExit(result.returncode)
