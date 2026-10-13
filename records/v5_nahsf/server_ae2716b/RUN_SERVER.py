"""V5三个种子并行运行；原V4控制组只读，不重训。"""
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
DEPLOYMENT = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())
STRONG = Path(DEPLOYMENT["strong_data_root"]) / "generated/target"


def save(path, value):
    """原子保存本轮状态，SSH断开仍可核对进度和失败返回码。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def worker(seed):
    """依次训练、原始八条件评估及5/0压力评估；失败不自动重试。"""
    commands = [
        ("train", ["train", "--epochs", "50", "--batch", "32", "--lr", "0.001"]),
        ("evaluate_original", ["evaluate", "--snr-root", str(ROOT / "target_snr")]),
        ("evaluate_stress", ["evaluate", "--strong-snr-root", str(STRONG), "--conditions", "5", "0"])]
    state = {"seed": seed, "status": "running", "stages": []}
    for name, args in commands:
        command = [sys.executable, "-u", "scripts/v5_experiment.py", *args, "--seeds", str(seed), "--output", str(OUT)]
        started = time.monotonic()
        log = ROOT / "logs" / f"seed{seed}_{name}.log"
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=CODE, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            state.update(stage=name, pid=process.pid, command=command)
            save(ROOT / "logs" / f"seed{seed}.json", state)
            try:
                exit_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_code = process.wait(timeout=30)
                state["hard_timeout"] = True
        state["stages"].append({"stage": name, "command": command, "exit_code": exit_code,
                                "elapsed_seconds": time.monotonic()-started})
        if exit_code or state.get("hard_timeout"):
            state["status"] = "failed"
            save(ROOT / "logs" / f"seed{seed}.json", state)
            return state
    state["status"] = "completed"
    save(ROOT / "logs" / f"seed{seed}.json", state)
    print(f"COMPLETED seed{seed}", flush=True)
    return state


def main():
    """每个进程四条CPU线程，三种子并行；结束后执行原V5配对汇总。"""
    assert not OUT.exists() and json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    (ROOT / "logs").mkdir()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        workers = list(pool.map(worker, (42, 43, 44)))
    if any(r["status"] != "completed" for r in workers):
        save(ROOT / "RUN_STATUS.json", {"status": "failed", "workers": workers})
        raise SystemExit(1)
    command = [sys.executable, "scripts/v5_experiment.py", "summarize", "--output", str(OUT),
               "--conditions", "clean", "10", "15", "20", "25", "30", "35", "40", "5", "0",
               "--v4-original-results", DEPLOYMENT["original_v4"] + "/results/evaluations",
               "--v4-stress-results", DEPLOYMENT["original_pressure_test"] + "/results/evaluations"]
    with (ROOT / "summary.log").open("x") as log:
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    save(ROOT / "RUN_STATUS.json", {"status": "completed" if result.returncode == 0 else "failed", "workers": workers,
          "summary_command": command, "summarize_exit_code": result.returncode, "elapsed_seconds": time.monotonic()-started,
          "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
