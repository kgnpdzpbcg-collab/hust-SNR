"""Standalone source-only plain CNN1D control, matching the frozen ResNet protocol.

Does not change historical SVM/RF/ResNet/V5/Xu training or results.
Only uses source labels for fitting/selection; Target-Dev strictly for reporting.

Run with the same data directories as scripts/run_classical_baselines_server.sh.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from plain_cnn1d_baseline import FivePhasePlainCNN1D
from v1_noise_consistency import fix_randomness, metrics, score_checkpoints
import v2_experiment as data
import classical_baseline_experiment as old

CONFIG = Path(__file__).resolve().parents[1] / "configs/plain_cnn1d_control_v1.json"
DEFAULT_OUT = Path("outputs/plain_cnn1d_control_v1")
SEEDS = (42,43,44)
CONDITIONS = (None,10,15,20,25,30,35,40,5,0)
METHOD = "cnn1d_plain"


def configuration():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    reference = old.config()
    assert c["reference_classical_protocol"] == reference["protocol"]
    assert c["seeds"] == list(SEEDS)
    assert c["target_conditions"] == ["clean",10,15,20,25,30,35,40,5,0]
    assert c["source_train_snr_db"] == reference["training_noise_levels_db"]
    assert c["source_val_snr_db"] == reference["validation_noise_levels_db"]
    assert c["epochs"] == reference["resnet1d"]["epochs"] == 50
    assert c["batch_size"] == reference["resnet1d"]["batch_size"] == 32
    assert c["learning_rate"] == reference["resnet1d"]["learning_rate"] == .001
    assert c["weight_decay"] == reference["resnet1d"]["weight_decay"] == 0.
    return c


def train(seed:int, out:Path, source_strong_root:Path):
    c=configuration()
    if seed not in SEEDS:raise ValueError("Seed outside 42,43,44")
    dst=out/"models"/METHOD/f"seed{seed}"
    if dst.exists():raise FileExistsError(f"Refusing to overwrite {dst}")
    fix_randomness(seed)
    (tr,tm,y,clean),(vr,vm,vy,clean_val)=old.source_views()
    assert len(y)==738 and len(vy)==302
    val=old.noise_source_val(source_strong_root,vr,vm)
    assert set(val)==set(c["source_val_snr_db"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FivePhasePlainCNN1D().to(device)
    init_hash=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                       for p in model.parameters())).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=c["learning_rate"],weight_decay=0.)
    history=[]
    states={}
    for epoch in range(1,c["epochs"]+1):
        noisy=data.fresh_noisy_source(tr,tm,seed,epoch)
        order=np.random.default_rng(seed+901+epoch).permutation(len(y))
        model.train()
        losses=[]
        for k in range(0,len(order),c["batch_size"]):
            ids=order[k:k+c["batch_size"]]
            x=torch.from_numpy(clean[ids]).to(device)
            xn=torch.from_numpy(noisy[ids]).to(device)
            yy=torch.from_numpy(y[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=old._resnet_loss(model,x,xn,yy)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"CNN1D loss nonfinite: seed={seed} epoch={epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        clean_f1=metrics(vy,old._tensor_predict(model,clean_val,device))["macro_f1"]
        noise_f1={str(level):metrics(vy,old._tensor_predict(model,v,device))["macro_f1"]
                  for level,v in val.items()}
        row=dict(epoch=epoch,train_loss_mean=float(np.mean(losses)),
                 clean_f1=float(clean_f1),noisy_f1=noise_f1,
                 noise_mean_f1=float(np.mean(list(noise_f1.values()))))
        history.append(row)
        states[epoch]={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        if epoch==1 or epoch%10==0 or epoch==c["epochs"]:
            print(f"{METHOD} seed{seed} epoch{epoch}/50 source Clean={clean_f1:.4f} "
                  f"5SNR={row['noise_mean_f1']:.4f}",flush=True)
    chosen=score_checkpoints(history)
    dst.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state=states[chosen["epoch"]],method=METHOD,seed=seed,
                    protocol=c["protocol"],source_only=True,selected=chosen,
                    final_holdout_evaluated=False),dst/"best.pt")
    old._write(dst/"history.json",history)
    old._write(dst/"run.json",dict(method=METHOD,seed=seed,protocol=c["protocol"],
                reference_classical_protocol=c["reference_classical_protocol"],
                source_split="34 train / 15 val records",selected=chosen,
                initial_weights_sha256=init_hash,
                parameters=sum(p.numel() for p in model.parameters()),
                train_epochs=c["epochs"], batch_size=c["batch_size"],
                source_only=True,target_used_for_training_or_selection=False,
                final_holdout_evaluated=False))


def evaluate(seed:int,out:Path,conditions,*,snr_root=None,strong_root=None,dev_export_root=None):
    c=configuration()
    folder=out/"models"/METHOD/f"seed{seed}"
    run=json.loads((folder/"run.json").read_text())
    if (run["method"],run["seed"],run["protocol"])!=(METHOD,seed,c["protocol"]):
        raise ValueError("Run provenance mismatch")
    if run["target_used_for_training_or_selection"] or run["final_holdout_evaluated"]:
        raise ValueError("Target leakage marker")
    ck=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    if (ck["method"],ck["seed"],ck["protocol"])!=(METHOD,seed,c["protocol"]):
        raise ValueError("Checkpoint provenance mismatch")
    fix_randomness(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FivePhasePlainCNN1D().to(device)
    model.load_state_dict(ck["state"],strict=True)
    from prepare import OUT as SOURCE_OUT,read
    metadata=read(SOURCE_OUT/"target_dev_clean_windows.json")
    truth=np.asarray([row["label"] for row in metadata],dtype=np.int64)
    if len(metadata)!=2474 or len({row["record_id"] for row in metadata})!=88:
        raise ValueError("Only frozen Target-Dev 88/2474 permitted")
    for snr in conditions:
        label="clean" if snr is None else f"snr{snr}"
        dst=out/"evaluations"/METHOD/f"seed{seed}"/f"target_dev_{label}.json"
        if dst.exists():raise FileExistsError(f"Refusing overwrite {dst}")
        raw,actual=old.target_raw(snr,snr_root=snr_root,strong_root=strong_root,
                                  dev_export_root=dev_export_root)
        if actual!=metadata:raise ValueError("Target Dev metadata changed")
        x=data.raw_cycles(raw,metadata)
        pred=np.asarray(old._tensor_predict(model,x,device),dtype=np.int64)
        score=metrics(truth,pred)
        old._write(dst,dict(method=METHOD,seed=seed,condition=label,
                protocol=c["protocol"],record_families=88,windows=2474,
                truth=truth.tolist(),predicted=pred.tolist(),metrics=score,
                selected_source=run["selected"],source_only_selection=True,
                synthetic_target_noise=snr in (0,5),final_holdout_evaluated=False))
        print(f"{METHOD} seed{seed} {label}: Macro-F1={100*score['macro_f1']:.3f}%",flush=True)


def summarize(out:Path,seeds,conditions):
    c=configuration()
    rows=[];per_class=[];stats=[]
    common_truth=None
    for seed in seeds:
        for snr in conditions:
            name="clean" if snr is None else f"snr{snr}"
            rec=json.loads((out/"evaluations"/METHOD/f"seed{seed}"/
                            f"target_dev_{name}.json").read_text())
            if (rec["method"],rec["seed"],rec["condition"],rec["protocol"])!=(METHOD,seed,name,c["protocol"]):
                raise ValueError("Wrong prediction provenance")
            if rec["record_families"]!=88 or rec["windows"]!=2474 or rec["final_holdout_evaluated"] is not False:
                raise ValueError("Unexpected Target-Dev cohort")
            if common_truth is None:common_truth=rec["truth"]
            if rec["truth"]!=common_truth:raise ValueError("Target-Dev cycle order changed")
            m=rec["metrics"]
            rows.append(dict(method=METHOD,seed=seed,condition=name,
                macro_f1_percent=100*m["macro_f1"],accuracy_percent=100*m["accuracy"],
                fault_as_healthy=m["missed_fault_as_normal"],
                healthy_as_fault=m["normal_false_alarms"]))
            for cls,(f1,recall) in enumerate(zip(m["per_class_f1"],m["per_class_recall"])):
                per_class.append(dict(method=METHOD,seed=seed,condition=name,label=cls,
                                f1_percent=100*f1,recall_percent=100*recall))
    for snr in conditions:
        name="clean" if snr is None else f"snr{snr}"
        sub=[x for x in rows if x["condition"]==name]
        f1=np.asarray([x["macro_f1_percent"] for x in sub])
        acc=np.asarray([x["accuracy_percent"] for x in sub])
        stats.append(dict(method=METHOD,condition=name,n_seeds=len(sub),
                          macro_f1_mean=float(f1.mean()),macro_f1_sd=float(f1.std(ddof=1)) if len(sub)>1 else 0.,
                          accuracy_mean=float(acc.mean()),accuracy_sd=float(acc.std(ddof=1)) if len(sub)>1 else 0.))
    report=out/"reports";report.mkdir(parents=True,exist_ok=True)
    old._csv(report/"PER_SEED_RESULTS.csv",rows)
    old._csv(report/"PER_CLASS_PER_SEED.csv",per_class)
    old._write(report/"SUMMARY.json",dict(protocol=c["protocol"],
        method=METHOD,seeds=list(seeds),target_dev_records=88,
        final_holdout_evaluated=False,summary=stats))
    print("Plain CNN1D reports:",report,flush=True)


def smoke():
    from plain_cnn1d_baseline import smoke as model_smoke
    model_smoke()
    assert configuration()["seeds"]==list(SEEDS)
    model=FivePhasePlainCNN1D()
    clean=torch.randn(4,5,4096)
    y=torch.tensor([0,1,2,3])
    loss=old._resnet_loss(model,clean,clean+.05*torch.randn_like(clean),y)
    loss.backward()
    assert torch.isfinite(loss) and model.classifier.weight.grad is not None
    print("PlainCNN1D protocol smoke PASS",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("smoke","train","evaluate","summarize"))
    p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    p.add_argument("--out",type=Path,default=DEFAULT_OUT)
    p.add_argument("--source-strong-root",type=Path)
    p.add_argument("--snr-root",type=Path)
    p.add_argument("--strong-root",type=Path)
    p.add_argument("--dev-export-root",type=Path)
    p.add_argument("--conditions",nargs="+",default=["clean","10","15","20","25","30","35","40","5","0"])
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or any(s not in SEEDS for s in a.seeds):
        p.error("Only frozen seeds 42,43,44 are allowed")
    cond=[None if z=="clean" else int(z) for z in a.conditions]
    if len(set(cond))!=len(cond) or any(x is not None and x not in CONDITIONS for x in cond):
        p.error("Invalid SNR values or duplicates")
    if a.stage=="smoke":smoke();return
    if a.stage=="train":
        if a.source_strong_root is None:p.error("--source-strong-root required")
        for seed in a.seeds:train(seed,a.out,a.source_strong_root)
    elif a.stage=="evaluate":
        if a.snr_root is None and a.dev_export_root is None:p.error("--snr-root or --dev-export-root required")
        if any(s in (0,5) for s in cond) and a.strong_root is None and a.dev_export_root is None:
            p.error("--strong-root required for 0/5 dB")
        for seed in a.seeds:
            evaluate(seed,a.out,cond,snr_root=a.snr_root,strong_root=a.strong_root,
                     dev_export_root=a.dev_export_root)
    else:summarize(a.out,a.seeds,cond)


if __name__=="__main__":main()
