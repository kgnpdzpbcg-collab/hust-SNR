"""在服务器后台启动控制器；所有子进程复用当前uv解释器和独立日志。"""
from pathlib import Path
import json, os, subprocess, sys
ROOT = Path(__file__).resolve().parent
assert not (ROOT / "CONTROLLER_PID.json").exists()
with (ROOT / "controller.log").open("x", encoding="utf-8") as log:
    process = subprocess.Popen([sys.executable, "-u", str(ROOT / "RUN_SERVER.py")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = {"pid":process.pid,"python":sys.executable,"cwd":str(ROOT)}
(ROOT / "CONTROLLER_PID.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record,ensure_ascii=False))
