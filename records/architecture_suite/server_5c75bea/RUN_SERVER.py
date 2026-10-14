"""四个工作进程批量执行27组原仓库九模型训练与270组评估，独立保存日志。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE,OUT = ROOT/'code',ROOT/'results'
DEPLOY = json.loads((ROOT/'DEPLOYMENT_MANIFEST.json').read_text())

def save(path,value):
    """状态记录原子更新，不改动正式模型及预测文件。"""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def worker(task):
    """单模型单种子先源训练、再十条件Dev推理；失败保留日志，不自动改配置。"""
    method,seed = task
    name = f'{method}_seed{seed}'
    state = {'method':method,'seed':seed,'status':'running','stages':[]}
    for stage in ('train','evaluate'):
        command = [sys.executable,'-u','scripts/hust_arch_suite_experiment.py',stage,
            '--methods',method,'--seeds',str(seed),'--out',str(OUT)]
        if stage == 'train': command += ['--source-strong-root',DEPLOY['source_strong_root']]
        else: command += ['--snr-root',DEPLOY['target_snr_root'],'--strong-root',DEPLOY['target_strong_root']]
        started = time.monotonic()
        with (ROOT/'logs'/f'{name}_{stage}.log').open('x') as log:
            process = subprocess.Popen(command,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=stage,pid=process.pid,command=command)
            save(ROOT/'logs'/f'{name}.json',state)
            try: exit_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_code = process.wait(timeout=30)
                state['hard_timeout'] = True
        state['stages'].append({'stage':stage,'command':command,'exit_code':exit_code,
            'elapsed_seconds':time.monotonic()-started})
        if exit_code or state.get('hard_timeout'):
            state['status'] = 'failed'
            save(ROOT/'logs'/f'{name}.json',state)
            return state
    state['status'] = 'completed'
    save(ROOT/'logs'/f'{name}.json',state)
    print('NINE_ARM_COMPLETED',name,flush=True)
    return state

def main():
    """最多四进程×四线程，复用同一GPU；所有27组成功后按原入口统一汇总。"""
    assert not OUT.exists() and json.loads((ROOT/'PREFLIGHT.json').read_text())['status'] == 'passed'
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    (ROOT/'logs').mkdir()
    started = time.monotonic()
    tasks = [(method,seed) for method in DEPLOY['config']['methods'] for seed in DEPLOY['config']['seeds']]
    with ThreadPoolExecutor(max_workers=4) as pool: workers = list(pool.map(worker,tasks))
    if any(state['status'] != 'completed' for state in workers):
        save(ROOT/'RUN_STATUS.json',{'status':'failed','workers':workers})
        raise SystemExit(1)
    command = [sys.executable,'scripts/hust_arch_suite_experiment.py','summarize','--out',str(OUT)]
    with (ROOT/'summary.log').open('x') as log:
        done = subprocess.run(command,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    save(ROOT/'RUN_STATUS.json',{'status':'completed' if done.returncode == 0 else 'failed','workers':workers,
        'summary_command':command,'elapsed_seconds':time.monotonic()-started})
    raise SystemExit(done.returncode)

if __name__ == '__main__': main()
