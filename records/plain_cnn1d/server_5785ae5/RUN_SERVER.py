"""同一服务器并发三个CNN1D种子，每个先源训练，再进行十条件评估。"""
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
    """只更新本轮进程状态，原子替换防止读取到半份JSON。"""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def worker(seed):
    """独立种子日志和结果目录；失败保留证据，不改变模型或噪声。"""
    state = {'seed':seed,'status':'running','stages':[]}
    for stage in ('train','evaluate'):
        command = [sys.executable,'-u','scripts/plain_cnn1d_experiment.py',stage,
            '--seeds',str(seed),'--out',str(OUT)]
        if stage == 'train': command += ['--source-strong-root',DEPLOY['source_strong_root']]
        else: command += ['--snr-root',DEPLOY['target_snr_root'],'--strong-root',DEPLOY['target_strong_root']]
        started = time.monotonic()
        with (ROOT/'logs'/f'seed{seed}_{stage}.log').open('x') as log:
            process = subprocess.Popen(command,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=stage,pid=process.pid,command=command)
            save(ROOT/'logs'/f'seed{seed}.json',state)
            try: exit_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_code = process.wait(timeout=30)
                state['hard_timeout'] = True
        state['stages'].append({'stage':stage,'command':command,'exit_code':exit_code,
            'elapsed_seconds':time.monotonic()-started})
        if exit_code or state.get('hard_timeout'):
            state['status'] = 'failed'
            save(ROOT/'logs'/f'seed{seed}.json',state)
            return state
    state['status'] = 'completed'
    save(ROOT/'logs'/f'seed{seed}.json',state)
    print('PLAIN_CNN1D_COMPLETED',seed,flush=True)
    return state

def main():
    """复用原环境，三进程各四线程；全部成功后执行原仓库汇总。"""
    assert not OUT.exists() and json.loads((ROOT/'PREFLIGHT.json').read_text())['status'] == 'passed'
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    (ROOT/'logs').mkdir()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool: workers = list(pool.map(worker,(42,43,44)))
    if any(state['status'] != 'completed' for state in workers):
        save(ROOT/'RUN_STATUS.json',{'status':'failed','workers':workers})
        raise SystemExit(1)
    command = [sys.executable,'scripts/plain_cnn1d_experiment.py','summarize','--out',str(OUT)]
    with (ROOT/'summary.log').open('x') as log:
        done = subprocess.run(command,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    save(ROOT/'RUN_STATUS.json',{'status':'completed' if done.returncode == 0 else 'failed','workers':workers,
        'summary_command':command,'elapsed_seconds':time.monotonic()-started})
    raise SystemExit(done.returncode)

if __name__ == '__main__': main()
