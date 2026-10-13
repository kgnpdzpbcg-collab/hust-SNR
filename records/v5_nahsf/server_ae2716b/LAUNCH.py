"""服务器独立启动控制器，SSH结束后训练和日志仍保留。"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
with (root / "controller.log").open("x") as log:
    process = subprocess.Popen([sys.executable, "-u", str(root / "RUN_SERVER.py")], cwd=root,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / "CONTROLLER_PID.json").write_text(json.dumps({"pid": process.pid, "python": sys.executable, "cwd": str(root)}, indent=2) + "\n")
print("V5_CONTROLLER_STARTED", process.pid)
