"""SSH关闭后仍运行九模型实验，本轮保留控制进程PID及输出。"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
assert not (root/'CONTROLLER_PID.json').exists()
with (root/'controller.log').open('x') as log:
    process = subprocess.Popen([sys.executable,'-u',str(root/'RUN_SERVER.py')],cwd=root,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'CONTROLLER_PID.json').write_text(json.dumps({'pid':process.pid,'python':sys.executable})+'\n')
print('NINE_ARM_CONTROLLER_STARTED',process.pid)
