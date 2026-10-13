"""串行提取GPU特征后，四CPU进程并行72组固定t-SNE，不做模型训练。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CODE=ROOT/"code"
OUT=ROOT/"results"
DEPLOY=json.loads((ROOT/"DEPLOYMENT_MANIFEST.json").read_text())


def main():
    """运行阶段失败保留日志，不改perplexity或筛数据重试。"""
    os.environ.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    assert not OUT.exists()
    started=time.monotonic();stages=[]
    for stage in ("extract","project"):
        command=[sys.executable,"-u","scripts/six_model_tsne.py",stage,"--output",str(OUT)]
        if stage=="extract":command += ["--base",DEPLOY["base"]]
        begin=time.monotonic()
        with (ROOT/f"{stage}.log").open("x") as log:
            process=subprocess.Popen(command,cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            state={"status":"running","stage":stage,"pid":process.pid,"command":command,"stages":stages}
            (ROOT/"RUN_STATUS.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n")
            try:code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                process.terminate();code=process.wait(timeout=30);state["hard_timeout"]=True
        stages.append({"stage":stage,"command":command,"exit_code":code,"elapsed_seconds":time.monotonic()-begin})
        if code or state.get("hard_timeout"):
            (ROOT/"RUN_STATUS.json").write_text(json.dumps({"status":"failed","stages":stages},ensure_ascii=False,indent=2)+"\n")
            raise SystemExit(1)
    (ROOT/"RUN_STATUS.json").write_text(json.dumps({"status":"completed","stages":stages,
        "elapsed_seconds":time.monotonic()-started},ensure_ascii=False,indent=2)+"\n")
    print("FEATURE_EXTRACTION_AND_TSNE_COMPLETED",flush=True)


if __name__=="__main__":main()
