"""在服务器启动独立控制器，SSH断开后仍保留运行和日志。"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
with (root / "controller.log").open("x") as log:
    process = subprocess.Popen([sys.executable, "-u", str(root / "RUN_SERVER.py")], cwd=root,
                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
record = {"pid": process.pid, "python": sys.executable, "cwd": str(root)}
(root / "CONTROLLER_PID.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
