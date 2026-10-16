"""单步执行并记录状态；由外部逐步核验后再调用下一步。"""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
TRAINING=ROOT.parents[1]/'source_cycle_benchmark/4728e9b'
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
stage=sys.argv[1]
assert stage in ['generate','verify-data','evaluate','verify-evaluation','extract','project']
if stage in ['extract','project']:
 command=[sys.executable,'-u',str(ROOT/'source_cycle_tsne.py'),stage,'--training-root',str(TRAINING),'--config',str(ROOT/'config.json'),'--output',str(ROOT/'tsne'),'--snr-root',str(ROOT/'data'),'--prediction-root',str(ROOT/'evaluation')]
else:
 command=[sys.executable,'-u',str(ROOT/'additional_snr_fixed_weights.py'),stage,'--training-root',str(TRAINING),'--config',str(ROOT/'config.json'),'--output',str(ROOT)]
state_path=ROOT/'STAGE_STATUS.json'
previous=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':[]}
started=time.monotonic()
with (ROOT/(stage+'.log')).open('x') as stream:
 result=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
job={'stage':stage,'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-started}
previous['jobs'].append(job)
previous.update(last_stage=stage,status='passed' if result.returncode==0 else 'failed')
state_path.write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(job,ensure_ascii=False))
raise SystemExit(result.returncode)
