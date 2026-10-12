"""四种方法并行，方法内依次训练三个种子并评估十个实测条件。"""
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
STRONG = Path(json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())["strong_data_root"])
METHODS = ["lowpass3", "lowpass4", "static4", "hpasf"]


def save(path, value):
    """状态原子更新，SSH查看时不会读到半份JSON。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def worker(method):
    """出错保留日志和失败状态，不自动重启正式训练。"""
    state = {"method": method, "status": "running", "stages": []}
    for stage in ("train", "evaluate"):
        command = [sys.executable, "-u", "scripts/v4_strong_noise_experiment.py", stage,
                   "--methods", method, "--output", str(OUT)]
        command += (["--strong-data-root", str(STRONG)] if stage == "train" else ["--snr-root", str(ROOT / "target_snr")])
        start = time.monotonic()
        log = ROOT / "logs" / f"{method}_{stage}.log"
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=CODE, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            state.update(stage=stage, pid=process.pid, command=command, log=str(log))
            save(ROOT / "logs" / f"{method}.json", state)
            try:
                exit_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_code = process.wait(timeout=30)
                state["hard_timeout"] = True
        state["stages"].append({"stage": stage, "command": command, "exit_code": exit_code,
                                "elapsed_seconds": time.monotonic() - start})
        if exit_code != 0 or state.get("hard_timeout"):
            state["status"] = "failed"
            save(ROOT / "logs" / f"{method}.json", state)
            return state
    state["status"] = "completed"
    save(ROOT / "logs" / f"{method}.json", state)
    print(f"COMPLETED {method}", flush=True)
    return state


def main():
    """复用原环境，四个进程各用四条CPU线程，结果只写本轮目录。"""
    assert not OUT.exists() and json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    (ROOT / "logs").mkdir()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        workers = list(pool.map(worker, METHODS))
    if any(w["status"] != "completed" for w in workers):
        save(ROOT / "RUN_STATUS.json", {"status": "failed", "workers": workers})
        raise SystemExit(1)
    command = [sys.executable, "scripts/v4_strong_noise_experiment.py", "summarize", "--output", str(OUT)]
    with (ROOT / "summary.log").open("x") as log:
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    save(ROOT / "RUN_STATUS.json", {"status": "completed" if result.returncode == 0 else "failed",
         "workers": workers, "summarize_exit_code": result.returncode, "elapsed_seconds": time.monotonic()-started,
         "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
