"""在服务器编排冻结入口：六进程并发训练，全部成功后统一测试与汇总。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

RUN = Path('/home/wang/projects/会议论文/hust/experiments/b1_paper_ablations/d9947f0_20261016')
CODE = RUN / 'code'
OUT = RUN / 'results'
MANIFEST = json.loads((RUN / 'DEPLOYMENT_MANIFEST.json').read_text())
CONFIG = json.loads((CODE / 'configs/b1_paper_ablations_v1.json').read_text())
ENV = os.environ.copy()
ENV.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
START = time.time()
STATUS = {'state': 'training', 'started_unix': START, 'jobs': [], 'workers': 6,
          'timeout_seconds': 1800, 'code_commit': MANIFEST['commit']}


def save_status():
    """以原子替换写状态；外部监测不会读到写到一半的JSON。"""
    STATUS['elapsed_seconds'] = time.time() - START
    temporary = RUN / 'PIPELINE_STATUS.tmp'
    temporary.write_text(json.dumps(STATUS, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temporary.replace(RUN / 'PIPELINE_STATUS.json')


def run_stage(stage, extra):
    """训练后只执行一次评价/汇总；任何非零返回均停止，不自动重跑。"""
    STATUS['state'] = stage
    save_status()
    command = [sys.executable, '-u', 'scripts/b1_paper_ablations.py', stage,
               '--out', str(OUT), '--source-root', MANIFEST['source_root'], *extra]
    stamp = time.time()
    with (RUN / 'logs' / f'{stage}.log').open('w') as log:
        result = subprocess.run(command, cwd=CODE, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                                timeout=max(1, 1800 - (time.time() - START)))
    STATUS[stage] = {'command': command, 'exit_code': result.returncode, 'seconds': time.time() - stamp}
    if result.returncode:
        raise RuntimeError(f'{stage} 失败；查看该阶段日志')


def main():
    """保留所有10组实际训练，不按运行成绩提前结束、筛选或增补种子。"""
    assert not OUT.exists(), '拒绝覆盖本轮已有结果'
    OUT.mkdir()
    jobs = [(m, s) for m in CONFIG['methods'] for s in CONFIG['seeds']]
    active = []
    save_status()
    while jobs or active:
        while jobs and len(active) < 6 and not STATUS.get('failed_job'):
            method, seed = jobs.pop(0)
            log = (RUN / 'logs' / f'{method}_seed{seed}_train.log').open('w')
            command = [sys.executable, '-u', 'scripts/b1_paper_ablations.py', 'train',
                       '--methods', method, '--seeds', str(seed), '--source-root', MANIFEST['source_root'],
                       '--out', str(OUT)]
            process = subprocess.Popen(command, cwd=CODE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
            row = {'method': method, 'seed': seed, 'pid': process.pid, 'command': command,
                   'started_unix': time.time(), 'state': 'running'}
            STATUS['jobs'].append(row)
            active.append((process, log, row))
            save_status()
        for process, log, row in active[:]:
            code = process.poll()
            if code is not None:
                log.close()
                row.update(exit_code=code, seconds=time.time() - row['started_unix'],
                           state='completed' if code == 0 else 'failed')
                active.remove((process, log, row))
                if code:
                    STATUS['failed_job'] = {'method': row['method'], 'seed': row['seed'], 'exit_code': code}
                save_status()
        if STATUS.get('failed_job') and not active:
            raise RuntimeError('训练失败；停止派发后续组，无自动重试')
        if time.time() - START > 1800:
            for process, log, row in active:
                process.terminate()
                log.close()
            raise TimeoutError('超过本轮30分钟运行上限')
        time.sleep(0.5)
    STATUS['training_seconds'] = time.time() - START
    run_stage('evaluate', ['--strong-root', MANIFEST['strong_root'], '--extreme-root', MANIFEST['extreme_root']])
    run_stage('summarize', [])
    STATUS['state'] = 'completed'
    save_status()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        STATUS.update(state='failed', error=repr(error))
        save_status()
        raise
