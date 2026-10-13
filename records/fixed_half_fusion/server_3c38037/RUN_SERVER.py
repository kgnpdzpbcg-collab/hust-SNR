"""三个种子并行，各自50轮训练及十条件评估；不重训或覆盖V5。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE, OUT = ROOT / "code", ROOT / "results"
DEPLOY = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())


def save(path, value):
    """原子更新本轮状态文件，便于SSH断开后查证运行状态。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def worker(seed):
    """失败即保留日志，不自动重跑、换选模规则或替换数据。"""
    state = {"seed": seed, "status": "running", "stages": []}
    for stage in ("train", "evaluate"):
        command = [sys.executable, "-u", "scripts/fixed_half_experiment.py", stage, "--seeds", str(seed),
                   "--output", str(OUT), "--strong-data-root", DEPLOY["strong_data_root"]]
        if stage == "evaluate":
            command += ["--snr-root", str(ROOT / "target_snr")]
        start = time.monotonic()
        with (ROOT / "logs" / f"seed{seed}_{stage}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            state.update(stage=stage, pid=process.pid, command=command)
            save(ROOT / "logs" / f"seed{seed}.json", state)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                code = process.wait(timeout=30)
                state["hard_timeout"] = True
        state["stages"].append({"stage": stage, "command": command, "exit_code": code,
                                "elapsed_seconds": time.monotonic()-start})
        if code or state.get("hard_timeout"):
            state["status"] = "failed"
            save(ROOT / "logs" / f"seed{seed}.json", state)
            return state
    state["status"] = "completed"
    save(ROOT / "logs" / f"seed{seed}.json", state)
    print(f"COMPLETED seed{seed}", flush=True)
    return state


def main():
    """三个独立进程各4CPU线程，复用PyTorch环境，同一GPU并发。"""
    assert not OUT.exists() and json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    (ROOT / "logs").mkdir()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        workers = list(pool.map(worker, (42, 43, 44)))
    if any(r["status"] != "completed" for r in workers):
        save(ROOT / "RUN_STATUS.json", {"status": "failed", "workers": workers})
        raise SystemExit(1)
    command = [sys.executable, "scripts/fixed_half_experiment.py", "summarize", "--output", str(OUT),
               "--v5-results", str(Path(DEPLOY["reference_root"]) / "results")]
    with (ROOT / "summary.log").open("x") as log:
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    save(ROOT / "RUN_STATUS.json", {"status": "completed" if result.returncode == 0 else "failed", "workers": workers,
          "summary_command": command, "elapsed_seconds": time.monotonic()-start,
          "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
