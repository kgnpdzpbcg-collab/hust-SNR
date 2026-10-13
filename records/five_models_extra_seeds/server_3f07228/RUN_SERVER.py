"""最多四个进程并发完成15个模型/种子任务，每个任务先训练再评估。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE, OUT = ROOT / "code", ROOT / "results"
DEPLOY = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())


def save(path, value):
    """状态文件原子更新，日志保留精确命令、进程号及阶段耗时。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def worker(task):
    """仅运行请求的模型/新种子；失败保留日志，不改配置或自动重试。"""
    method, seed = task
    state = {"method": method, "seed": seed, "status": "running", "stages": []}
    name = f"{method}_seed{seed}"
    for stage in ("train", "evaluate"):
        command = [sys.executable, "-u", "scripts/five_models_extra_seeds.py", stage,
            "--methods", method, "--seeds", str(seed), "--output", str(OUT), "--strong-data-root", DEPLOY["strong_data_root"]]
        if stage == "evaluate":
            command += ["--snr-root", str(ROOT / "target_snr")]
        start = time.monotonic()
        with (ROOT / "logs" / f"{name}_{stage}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            state.update(stage=stage, pid=process.pid, command=command)
            save(ROOT / "logs" / f"{name}.json", state)
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
            save(ROOT / "logs" / f"{name}.json", state)
            return state
    state["status"] = "completed"
    save(ROOT / "logs" / f"{name}.json", state)
    print(f"COMPLETED {name}", flush=True)
    return state


def main():
    """复用原PyTorch环境，四个进程各四条CPU线程，全部完成后合并六种子。"""
    assert not OUT.exists() and json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    (ROOT / "logs").mkdir()
    start = time.monotonic()
    tasks = list(itertools.product(DEPLOY["protocol"]["methods"], DEPLOY["protocol"]["seeds"]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        workers = list(pool.map(worker, tasks))
    if any(r["status"] != "completed" for r in workers):
        save(ROOT / "RUN_STATUS.json", {"status": "failed", "workers": workers})
        raise SystemExit(1)
    command = [sys.executable, "scripts/five_models_extra_seeds.py", "summarize", "--output", str(OUT),
        "--four-results", str(Path(DEPLOY["historical_four_root"]) / "results"),
        "--fixed-results", str(Path(DEPLOY["historical_fixed_root"]) / "results")]
    with (ROOT / "summary.log").open("x") as log:
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    save(ROOT / "RUN_STATUS.json", {"status": "completed" if result.returncode == 0 else "failed", "workers": workers,
          "summary_command": command, "elapsed_seconds": time.monotonic()-start,
          "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
