"""启动脱离SSH会话的控制进程；只运行本轮独立目录，不覆盖历史实验。"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
assert not (root / "CONTROLLER_PID.json").exists()
with (root / "controller.log").open("x", encoding="utf-8") as log:
    process = subprocess.Popen([sys.executable, "-u", str(root / "RUN_SERVER.py")], cwd=root,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / "CONTROLLER_PID.json").write_text(json.dumps({"pid": process.pid, "python": sys.executable}) + "\n", encoding="utf-8")
print("A0_B0_CONTROLLER_STARTED", process.pid)
