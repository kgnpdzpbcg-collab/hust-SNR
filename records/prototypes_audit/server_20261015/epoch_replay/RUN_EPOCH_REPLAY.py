"""六组原配置确定性重放；只补存终轮，原正式记录保持只读。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
REF = BASE / "experiments/prototypes_a0_b0/c0b30c9"
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")


def save(name, value):
    """状态写在本轮目录；不覆盖原实验输出。"""
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def job(method, seed):
    """每个进程调用冻结原train函数，核对全部50轮历史和原最佳参数。"""
    started = time.perf_counter()
    name = f"{method}_seed{seed}"
    command = [sys.executable, "-u", str(ROOT / "audit_a0_b0_epoch_replay.py"),
        "--reference-root", str(REF), "--output", str(ROOT / "results"), "--method", method, "--seed", str(seed)]
    print(f"START {name}", flush=True)
    with (ROOT / "logs" / f"{name}.log").open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=REF / "code", env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
    record = {"method": method, "seed": seed, "command": command, "exit_code": result.returncode,
              "elapsed_seconds": time.perf_counter() - started}
    save(f"logs/{name}.json", record)
    print(f"FINISHED {name} exit={result.returncode}", flush=True)
    if result.returncode:
        print((ROOT / "logs" / f"{name}.log").read_text(), flush=True)
        raise RuntimeError(name)
    return record


def main():
    """分模型3进程并发，预先只指定原best和epoch50两个比较点。"""
    assert not (ROOT / "results").exists() and not (ROOT / "logs").exists()
    (ROOT / "logs").mkdir()
    started, jobs = time.perf_counter(), []
    try:
        for method in ("b0", "a0"):
            save("RUN_STATUS.json", {"status": "running", "method": method, "completed_jobs": len(jobs)})
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(job, method, seed) for seed in (42, 43, 44)]
                for future in as_completed(futures):
                    jobs.append(future.result())
        save("RUN_STATUS.json", {"status": "completed", "jobs": jobs,
              "elapsed_seconds": time.perf_counter() - started, "final_holdout_accessed": False})
    except Exception as error:
        save("RUN_STATUS.json", {"status": "failed", "error": str(error), "jobs": jobs})
        raise


if __name__ == "__main__":
    main()
