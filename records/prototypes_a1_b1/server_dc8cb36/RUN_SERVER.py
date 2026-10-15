"""B1三种子并发完成后再运行A1三种子；全部源训练完成后才开始目标评估。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE, OUT = ROOT / "code", ROOT / "results"
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")


def status(value):
    """只更新本轮控制状态；各模型完整输出始终保存到独立日志。"""
    (ROOT / "RUN_STATUS.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command, name):
    """执行分支原CLI，记录命令及用时，失败保留证据并中止后续阶段。"""
    started = time.perf_counter()
    print("START", name, flush=True)
    with (ROOT / "logs" / f"{name}.log").open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
    record = {"name": name, "command": command, "cwd": str(CODE), "exit_code": result.returncode,
              "elapsed_seconds": time.perf_counter() - started}
    (ROOT / "logs" / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINISHED", name, "exit", result.returncode, "seconds", round(record["elapsed_seconds"], 3), flush=True)
    if result.returncode:
        print((ROOT / "logs" / f"{name}.log").read_text(encoding="utf-8"), flush=True)
        raise RuntimeError(name)
    return record


def main():
    """使用3个独立Python进程训练，每个进程保留原随机数和确定性CUDA设置。"""
    assert not OUT.exists()
    assert json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    (ROOT / "logs").mkdir()
    deployment = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())
    jobs, started = [], time.perf_counter()
    try:
        for method in ("b1", "a1"):
            status({"status": "running", "phase": "source_train", "method": method, "completed_jobs": len(jobs)})
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(run, [sys.executable, "-u", "scripts/hust_a1_b1_experiment.py", "train",
                           "--methods", method, "--seeds", str(seed), "--out", str(OUT),
                           "--source-strong-root", deployment["source_strong_root"]], f"{method}_seed{seed}_train")
                           for seed in (42, 43, 44)]
                for future in as_completed(futures):
                    jobs.append(future.result())
        # 目标推理只在六份源域选中权重冻结后启动，不参与任何训练或选择。
        status({"status": "running", "phase": "target_dev_evaluation", "completed_jobs": len(jobs)})
        jobs.append(run([sys.executable, "-u", "scripts/hust_a1_b1_experiment.py", "evaluate",
             "--methods", "b1", "a1", "--out", str(OUT),
             "--strong-root", deployment["target_strong_root"], "--extreme-root", deployment["target_extreme_root"],
             "--conditions", "5", "0", "-5", "-10"], "target_dev_evaluation"))
        jobs.append(run([sys.executable, "-u", "scripts/hust_a1_b1_experiment.py", "summarize",
             "--methods", "b1", "a1", "--out", str(OUT), "--conditions", "5", "0", "-5", "-10"], "summary"))
        status({"status": "completed", "jobs": jobs, "elapsed_seconds": time.perf_counter() - started,
                "training_jobs": 6, "target_dev_evaluation_sets": 24, "final_holdout_evaluated": False})
        print("ALL_TRAINING_AND_EVALUATION_COMPLETE", flush=True)
    except Exception as error:
        status({"status": "failed", "error": str(error), "completed_jobs": jobs,
                "elapsed_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
