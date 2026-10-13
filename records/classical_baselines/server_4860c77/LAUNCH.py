"""断开SSH后保持控制进程运行，PID和控制日志仅保存到本轮目录。"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
assert not (root/'CONTROLLER_PID.json').exists()
with (root/'controller.log').open('x') as log:
    process = subprocess.Popen([sys.executable, '-u', str(root/'RUN_SERVER.py')],
        cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
(root/'CONTROLLER_PID.json').write_text(json.dumps({'pid': process.pid, 'python': sys.executable})+'\n')
print('CLASSICAL_CONTROLLER_STARTED', process.pid)
