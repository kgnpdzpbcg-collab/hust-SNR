"""调度42组独立源训练；所有best冻结后统一评价12条件并汇总。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE, OUT = ROOT / "code", ROOT / "results"
os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")


def status(value):
    """由单一主线程更新状态，子任务日志各自保存。"""
    (ROOT / "RUN_STATUS.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command, name):
    """执行已冻结CLI；失败或超时保留日志，不自动重试、更换种子或参数。"""
    started = time.perf_counter()
    print("START", name, flush=True)
    with (ROOT / "logs" / f"{name}.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=log,
                                stderr=subprocess.STDOUT, timeout=1800)
    record = {"name": name, "command": command, "exit_code": result.returncode,
              "elapsed_seconds": time.perf_counter() - started}
    (ROOT / "logs" / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINISHED", name, "exit", result.returncode, "seconds", round(record["elapsed_seconds"], 3), flush=True)
    if result.returncode:
        raise RuntimeError(f"{name}: exit {result.returncode}")
    return record


def main():
    """单张GPU上6个独立进程；各进程使用2个CPU线程及原确定性设置。"""
    assert not OUT.exists() and not (ROOT / "logs").exists()
    assert json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    deploy = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())
    config = deploy["config"]
    (ROOT / "logs").mkdir()
    jobs, started = [], time.perf_counter()
    cli = [sys.executable, "-u", "scripts/source_cycle_benchmark.py"]
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = []
            for method in config["methods"]:
                for seed in config["seeds"]:
                    command = [*cli, "train", "--methods", method, "--seeds", str(seed),
                               "--source-root", deploy["source_root"], "--out", str(OUT)]
                    futures.append(pool.submit(run, command, f"{method}_seed{seed}_train"))
            status({"status": "running", "phase": "source_training", "training_jobs": 42, "completed_jobs": 0})
            for future in as_completed(futures):
                jobs.append(future.result())
                status({"status": "running", "phase": "source_training", "training_jobs": 42,
                        "completed_jobs": len(jobs), "elapsed_seconds": time.perf_counter() - started})
        status({"status": "running", "phase": "target_dev_evaluation", "completed_training_jobs": 42})
        jobs.append(run([*cli, "evaluate", "--source-root", deploy["source_root"], "--snr-root", deploy["snr_root"],
                         "--strong-root", deploy["strong_root"], "--extreme-root", deploy["extreme_root"],
                         "--out", str(OUT)], "target_dev_evaluation"))
        jobs.append(run([*cli, "summarize", "--out", str(OUT)], "summary"))
        status({"status": "completed", "training_jobs": 42, "target_evaluation_sets": 504,
                "elapsed_seconds": time.perf_counter() - started, "jobs": jobs,
                "final_holdout_evaluated": False})
        print("ALL_42_TRAINING_AND_504_EVALUATIONS_COMPLETED", flush=True)
    except Exception as error:
        status({"status": "failed", "error": str(error), "completed_jobs": jobs,
                "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
