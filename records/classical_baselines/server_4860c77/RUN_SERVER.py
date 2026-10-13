"""最多四个独立进程并发执行九组基线训练及评估，保留实际种子和日志。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE, OUT = ROOT/'code', ROOT/'results'
DEPLOY = json.loads((ROOT/'DEPLOYMENT_MANIFEST.json').read_text())

def save(path, value):
    """进程状态原子替换，记录本轮路径，不修改模型目录。"""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)

def worker(task):
    """每组先完成源域训练和选模，再进行十条件Dev推理；失败不自动换协议。"""
    method, seed = task
    name = f'{method}_seed{seed}'
    state = {'method': method, 'seed': seed, 'status': 'running', 'stages': []}
    for stage in ('train', 'evaluate'):
        command = [sys.executable, '-u', 'scripts/classical_baseline_experiment.py', stage,
            '--methods', method, '--seeds', str(seed), '--out', str(OUT)]
        if stage == 'train': command += ['--source-strong-root', DEPLOY['source_strong_root']]
        else: command += ['--snr-root', DEPLOY['target_snr_root'],
            '--strong-root', DEPLOY['target_strong_root']]
        started = time.monotonic()
        with (ROOT/'logs'/f'{name}_{stage}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=CODE, env=os.environ.copy(),
                stdout=log, stderr=subprocess.STDOUT)
            state.update(stage=stage, pid=process.pid, command=command)
            save(ROOT/'logs'/f'{name}.json', state)
            try: exit_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_code = process.wait(timeout=30)
                state['hard_timeout'] = True
        state['stages'].append({'stage': stage, 'command': command,
            'exit_code': exit_code, 'elapsed_seconds': time.monotonic()-started})
        if exit_code or state.get('hard_timeout'):
            state['status'] = 'failed'
            save(ROOT/'logs'/f'{name}.json', state)
            return state
    state['status'] = 'completed'
    save(ROOT/'logs'/f'{name}.json', state)
    print('BASELINE_COMPLETED', name, flush=True)
    return state

def main():
    """四进程×四线程，三组ResNet先进入GPU队列，同时给CPU基线分配资源。"""
    assert not OUT.exists() and json.loads((ROOT/'PREFLIGHT.json').read_text())['status'] == 'passed'
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    (ROOT/'logs').mkdir()
    started = time.monotonic()
    tasks = [(method, seed) for method in ('resnet1d', 'svm_rbf', 'random_forest') for seed in (42,43,44)]
    with ThreadPoolExecutor(max_workers=4) as pool: workers = list(pool.map(worker, tasks))
    assert all(worker['status'] == 'completed' for worker in workers), workers
    command = [sys.executable, 'scripts/classical_baseline_experiment.py', 'summarize', '--out', str(OUT)]
    with (ROOT/'summary.log').open('x') as log:
        done = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    save(ROOT/'RUN_STATUS.json', {'status': 'completed' if done.returncode == 0 else 'failed',
        'workers': workers, 'summary_command': command, 'elapsed_seconds': time.monotonic()-started})
    raise SystemExit(done.returncode)

if __name__ == '__main__': main()
