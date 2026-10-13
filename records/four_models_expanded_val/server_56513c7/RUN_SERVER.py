"""四模型并行，各自完成三种子原训练、扩大验证及实测评估。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
OUT = ROOT / "results"
DEPLOY = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())
METHODS = DEPLOY["protocol"]["methods"]


def save(path, value):
    """原子更新进度状态，日志只写本轮目录。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def worker(method):
    """阶段失败保留错误日志，不自动重试或替换配置。"""
    state = {"method": method, "status": "running", "stages": []}
    for stage in ("train", "evaluate"):
        command = [sys.executable, "-u", "scripts/four_models_expanded_val.py", stage, "--methods", method,
                   "--output", str(OUT), "--strong-data-root", DEPLOY["strong_data_root"]]
        if stage == "evaluate":
            command += ["--snr-root", str(ROOT / "target_snr")]
        start = time.monotonic()
        with (ROOT / "logs" / f"{method}_{stage}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            state.update(stage=stage, pid=process.pid, command=command)
            save(ROOT / "logs" / f"{method}.json", state)
            try:
                exit_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_code = process.wait(timeout=30)
                state["hard_timeout"] = True
        state["stages"].append({"stage": stage, "command": command, "exit_code": exit_code,
                                "elapsed_seconds": time.monotonic()-start})
        if exit_code or state.get("hard_timeout"):
            state["status"] = "failed"
            save(ROOT / "logs" / f"{method}.json", state)
            return state
    state["status"] = "completed"
    save(ROOT / "logs" / f"{method}.json", state)
    print(f"COMPLETED {method}", flush=True)
    return state


def main():
    """复用原环境，4个进程各4条CPU线程；全部成功后才汇总。"""
    assert not OUT.exists() and json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    (ROOT / "logs").mkdir()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        workers = list(pool.map(worker, METHODS))
    if any(r["status"] != "completed" for r in workers):
        save(ROOT / "RUN_STATUS.json", {"status": "failed", "workers": workers})
        raise SystemExit(1)
    command = [sys.executable, "scripts/four_models_expanded_val.py", "summarize", "--output", str(OUT)]
    with (ROOT / "summary.log").open("x") as log:
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    save(ROOT / "RUN_STATUS.json", {"status": "completed" if result.returncode == 0 else "failed", "workers": workers,
          "summary_command": command, "elapsed_seconds": time.monotonic()-start,
          "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
