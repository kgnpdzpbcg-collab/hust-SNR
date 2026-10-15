"""使用原uv解释器后台启动控制器，断开SSH后继续运行并保存PID。"""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
assert not (ROOT / "CONTROLLER_PID.json").exists()
with (ROOT / "controller.log").open("x", encoding="utf-8") as stream:
    process = subprocess.Popen([sys.executable, "-u", str(ROOT / "RUN_SERVER.py")], cwd=ROOT,
                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / "CONTROLLER_PID.json").write_text(json.dumps({"pid": process.pid, "python": sys.executable,
    "cwd": str(ROOT)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("CONTROLLER_PID", process.pid)
