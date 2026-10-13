"""三个种子并行，只运行原仓库Xu-Full matched入口，所有命令明确保存。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CODE,OUT=ROOT/"code",ROOT/"results"
DEPLOY=json.loads((ROOT/"DEPLOYMENT_MANIFEST.json").read_text())


def save(path,value):
    """原子保存进程状态和耗时，独立日志不覆盖旧实验。"""
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(path)


def worker(seed):
    """每个种子先50轮训练，再十条件推理；失败保留日志，不自动重试。"""
    state={"mode":"xu_full","seed":seed,"status":"running","stages":[]}
    for stage in ("train","evaluate"):
        command=[sys.executable,"-u","scripts/xu2025_full_experiment.py",stage,"--track","matched",
                 "--modes","xu_full","--seeds",str(seed),"--out",str(OUT)]
        if stage=="train":
            command += ["--source-strong-root",DEPLOY["source_strong_root"],"--epochs","50","--batch","32",
                        "--lr","0.001","--weight-decay","0","--reconstruction","paper_sum","--rec-weight","1","--dml-weight","1"]
        else:
            command += ["--snr-root",str(ROOT/"target_snr"),"--strong-root",DEPLOY["target_strong_root"],
                        "--conditions","clean","10","15","20","25","30","35","40","5","0"]
        start=time.monotonic()
        with (ROOT/"logs"/f"seed{seed}_{stage}.log").open("x") as log:
            process=subprocess.Popen(command,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=stage,pid=process.pid,command=command)
            save(ROOT/"logs"/f"seed{seed}.json",state)
            try:
                exit_code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate()
                exit_code=process.wait(timeout=30)
                state["hard_timeout"]=True
        state["stages"].append({"stage":stage,"command":command,"exit_code":exit_code,"elapsed_seconds":time.monotonic()-start})
        if exit_code or state.get("hard_timeout"):
            state["status"]="failed"
            save(ROOT/"logs"/f"seed{seed}.json",state)
            return state
    state["status"]="completed"
    save(ROOT/"logs"/f"seed{seed}.json",state)
    print("COMPLETED Xu-Full seed",seed,flush=True)
    return state


def main():
    """复用旧环境，每进程4CPU线程，同一GPU跑三个种子；全部完成后汇总。"""
    assert not OUT.exists() and json.loads((ROOT/"PREFLIGHT.json").read_text())["status"]=="passed"
    os.environ.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    (ROOT/"logs").mkdir()
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        workers=list(pool.map(worker,(42,43,44)))
    if any(w["status"]!="completed" for w in workers):
        save(ROOT/"RUN_STATUS.json",{"status":"failed","workers":workers})
        raise SystemExit(1)
    command=[sys.executable,"scripts/xu2025_full_experiment.py","summarize","--track","matched","--modes","xu_full",
             "--seeds","42","43","44","--out",str(OUT)]
    with (ROOT/"summary.log").open("x") as log:
        done=subprocess.run(command,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
    save(ROOT/"RUN_STATUS.json",{"status":"completed" if done.returncode==0 else "failed","workers":workers,
         "summary_command":command,"elapsed_seconds":time.monotonic()-start,"finished_at_utc":datetime.now(timezone.utc).isoformat()})
    raise SystemExit(done.returncode)


if __name__=="__main__":main()
