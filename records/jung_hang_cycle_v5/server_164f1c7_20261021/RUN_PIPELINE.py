"""服务器四组训练并发编排，成功后顺序进行固定权重测试、核验及汇总。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

RUN=Path('/home/wang/projects/会议论文/hust/experiments/jung_hang_cycle_v5/164f1c7_20261021')
M=json.loads((RUN/'DEPLOYMENT_MANIFEST.json').read_text())
OUT=RUN/'results'
ENV=os.environ.copy();ENV.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
START=time.time()
STATE={'state':'training','started_unix':START,'jobs':[],'workers':4,'timeout_seconds':1800}


def save():
    STATE['elapsed_seconds']=time.time()-START
    p=RUN/'PIPELINE_STATUS.tmp';p.write_text(json.dumps(STATE,ensure_ascii=False,indent=2)+'\n',encoding='utf8');p.replace(RUN/'PIPELINE_STATUS.json')


def command(stage):
    return [sys.executable,'-u',str(RUN/'code/scripts/jung_hang_cycle_v5.py'),stage,'--config',str(RUN/'code/configs/jung_hang_v5_v1.json'),'--reference-root',M['reference_root'],
        '--source-root',M['source_root'],'--additional-root',M['additional_root'],'--previous-output',M['previous_output'],'--output',str(OUT)]


def main():
    assert OUT.exists() and sorted(p.name for p in OUT.iterdir())==['PREFLIGHT_PAPER_AUDIT.json'],'只允许预检记录存在，拒绝覆盖训练结果'
    active=[];save()
    for method in ('jung','hang'):
        for seed in (42,44):
            log=(RUN/f'logs/{method}_seed{seed}_train.log').open('w')
            cmd=command('train')+['--method',method,'--seed',str(seed)]
            p=subprocess.Popen(cmd,cwd=RUN/'code',env=ENV,stdout=log,stderr=subprocess.STDOUT)
            row={'method':method,'seed':seed,'pid':p.pid,'command':cmd,'started_unix':time.time(),'state':'running'}
            STATE['jobs'].append(row);active.append((p,log,row));save()
    while active:
        for p,log,row in active[:]:
            code=p.poll()
            if code is not None:
                log.close();row.update(exit_code=code,state='completed' if code==0 else 'failed',seconds=time.time()-row['started_unix'])
                active.remove((p,log,row));save()
        if time.time()-START>1800:
            for p,log,row in active:p.terminate();log.close()
            raise TimeoutError('超过30分钟运行上限')
        time.sleep(0.5)
    if any(r['exit_code'] for r in STATE['jobs']):raise RuntimeError('有训练组失败，不自动重试')
    STATE['training_seconds']=time.time()-START
    for stage in ('evaluate','verify','summarize'):
        STATE['state']=stage;save();stamp=time.time();cmd=command(stage)
        with (RUN/f'logs/{stage}.log').open('w') as log:
            r=subprocess.run(cmd,cwd=RUN/'code',env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,1800-(time.time()-START)))
        STATE[stage]={'command':cmd,'exit_code':r.returncode,'seconds':time.time()-stamp};save()
        if r.returncode:raise RuntimeError(f'{stage}失败，不自动重试')
    STATE['state']='completed';save()


if __name__=='__main__':
    try:main()
    except Exception as e:
        STATE.update(state='failed',error=repr(e));save();raise
