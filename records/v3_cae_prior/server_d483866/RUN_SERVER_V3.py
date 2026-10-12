"""并发调用冻结V3的四个CLI，不修改模型、输入、损失或选模规则。

每组依次完成三个种子的训练与原MAT测评；全部成功后调用原汇总入口。
状态、日志及返回码只写本轮目录，便于SSH断开后继续核对。
"""
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
SNR = ROOT / "target_snr"
METHODS = ["cae_identity", "cae_lowpass3", "cae_prior", "cae_prior_no_rec"]


def save(path, value):
    """原子更新本轮状态文件，读取方不会看到半份JSON。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def worker(method):
    """每种方法调用原脚本的train/evaluate，保留每阶段返回码及耗时。"""
    state = {"method": method, "status": "running", "stages": [],
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    state_file = ROOT / "logs" / f"{method}.json"
    save(state_file, state)
    for stage in ["train", "evaluate"]:
        command = [sys.executable, "-u", "scripts/v3_experiment.py", stage,
                   "--methods", method, "--seeds", "42", "43", "44",
                   "--output", str(OUT)]
        if stage == "train":
            command += ["--epochs", "50", "--batch", "32", "--lr", "0.001"]
        else:
            command += ["--snr-root", str(SNR)]
        log_path = ROOT / "logs" / f"{method}_{stage}.log"
        start = time.monotonic()
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=CODE, stdout=log,
                                       stderr=subprocess.STDOUT, env=os.environ.copy())
            state.update(stage=stage, pid=process.pid, command=command, log=str(log_path))
            save(state_file, state)
            # 每个CLI独立设30分钟硬超时；状态文件保留异常，不重试失败训练。
            try:
                exit_code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                print(f"HARD TIMEOUT: {method} {stage}; terminating PID {process.pid}", flush=True)
                process.terminate()
                exit_code = process.wait(timeout=30)
                state["hard_timeout"] = True
        state["stages"].append({"stage": stage, "command": command,
                                "exit_code": exit_code, "elapsed_seconds": time.monotonic() - start})
        if exit_code != 0 or state.get("hard_timeout"):
            state["status"] = "failed"
            save(state_file, state)
            return state
    state.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
    save(state_file, state)
    return state


def main():
    """只在空结果目录启动四组并发，全部成功后生成原V3汇总。"""
    assert not OUT.exists(), "不能覆盖已有V3结果"
    assert json.loads((ROOT / "PREFLIGHT.json").read_text())["status"] == "passed"
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    (ROOT / "logs").mkdir(exist_ok=False)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, METHODS))
    if any(item["status"] != "completed" for item in results):
        save(ROOT / "RUN_STATUS.json", {"status": "failed", "workers": results})
        raise SystemExit(1)
    command = [sys.executable, "scripts/v3_experiment.py", "summarize", "--output", str(OUT)]
    with (ROOT / "summary.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, check=False)
    save(ROOT / "RUN_STATUS.json", {"status": "completed" if result.returncode == 0 else "failed",
                                   "workers": results, "summarize_exit_code": result.returncode,
                                   "elapsed_seconds": time.monotonic() - start,
                                   "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
