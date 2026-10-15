"""SVM/RF/ResNet1D, strict source-only three-seed HUST baseline study.

One NEW protocol, no changes to previous V5/Xu baselines. Shared source
34/15 record split, frozen 88-record Target-Dev, synthetic 0/5 dB clearly
flagged, no target labels or holdout involved in fit/selection.

Commands (repo root):
 python scripts/classical_baseline_experiment.py smoke
 python scripts/classical_baseline_experiment.py train --source-strong-root /root/strong
 python scripts/classical_baseline_experiment.py evaluate --snr-root /root/SNR \
    --strong-root /root/strong
 python scripts/classical_baseline_experiment.py summarize
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from classical_baseline_models import FivePhaseResNet1D, engineered_features, FEATURE_VERSION
from v1_noise_consistency import fix_randomness, metrics, score_checkpoints
import v2_experiment as data

METHODS = ("svm_rbf","random_forest","resnet1d")
DEFAULT_CONDITIONS = ("clean",10,15,20,25,30,35,40,5,0)
CONFIG = Path(__file__).resolve().parents[1] / "configs/classical_baselines_v1.json"
OUT = Path("outputs/classical_baselines_v1")


def config() -> dict:
    v = json.loads(CONFIG.read_text())
    expected = dict(protocol="HUST-classical-baselines-source-only-v1",
                    methods=list(METHODS),seeds=[42,43,44],
                    validation_noise_levels_db=[0,5,10,20,30],
                    target_conditions=list(DEFAULT_CONDITIONS),
                    training_noise_levels_db=[10,15,20,25,30,35,40])
    if any(v.get(k)!=value for k,value in expected.items()):
        raise ValueError("Classical baselines frozen protocol changed")
    if v["resnet1d"]["epochs"]!=50 or v["resnet1d"]["batch_size"]!=32:
        raise ValueError("ResNet must use V5 50-epoch batch32 budget")
    if v["classical"]["augmented_source_epochs"]!=7:
        raise ValueError("Source augmentation views must be frozen at 7")
    return v


def noise_source_val(source_strong_root: Path,raw,meta):
    if source_strong_root is None:
        raise ValueError("Source0/5 fixed validation root required")
    # Same API as the existing 4-model experiment.
    return data.fixed_noisy_validation(raw,meta,
        noise_levels=tuple(config()["validation_noise_levels_db"]),
        materialized_root=source_strong_root / "generated/source")


def source_views():
    tr,tm,y = data.source_data("train")
    vr,vm,vy = data.source_data("val")
    return (tr,tm,y,data.raw_cycles(tr,tm)),(vr,vm,vy,data.raw_cycles(vr,vm))


def _selection(rows):
    """Clean <= 1pp below best, then max noisy 0/5/10/20/30 mean F1.

    Classical model candidates stand in for neural epochs; no Target-Dev scores
    or features are consulted. source Clean and fixed noisy VAL only.
    """
    tol = .01
    top = max(item["clean_f1"] for item in rows)
    eligible = [r for r in rows if r["clean_f1"] >= top-tol-1e-12]
    return max(eligible,key=lambda r:(r["noise_mean_f1"],r["clean_f1"],-r["candidate_index"]))


def classical_candidates(method:str,seed:int):
    """Minimal source-only preregistered search. No test-set tuning."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    if method=="svm_rbf":
        for C in config()["classical"]["svm_C"]:
            yield {"C":float(C),"gamma":"scale"},Pipeline([
                ("scaler",StandardScaler()),
                ("clf",SVC(kernel="rbf",C=float(C),gamma="scale",cache_size=1024))])
    elif method=="random_forest":
        for leaf in config()["classical"]["rf_min_samples_leaf"]:
            opt={"n_estimators":int(config()["classical"]["rf_n_estimators"]),
                 "min_samples_leaf":int(leaf),"max_features":"sqrt"}
            yield opt,RandomForestClassifier(**opt,random_state=seed,n_jobs=4)
    else:
        raise ValueError(f"Invalid classical method: {method}")


def _classical_training_arrays(seed:int,tr,tm,y,clean):
    blocks=[engineered_features(clean)]
    # Seeds change the deterministic source ONLY training noise realization;
    # every candidate of the same seed receives byte-identical features.
    for epoch in range(1,config()["classical"]["augmented_source_epochs"]+1):
        noisy=data.fresh_noisy_source(tr,tm,seed,epoch)
        blocks.append(engineered_features(noisy))
    return np.concatenate(blocks),np.tile(y,len(blocks))


def train_classical(method:str,seed:int,out:Path,source_strong_root:Path):
    import joblib
    if method not in METHODS[:2]:raise ValueError(method)
    dst=out/"models"/method/f"seed{seed}"
    if dst.exists():raise FileExistsError(f"Refusing overwrite {dst}")
    fix_randomness(seed)
    (tr,tm,y,clean),(vr,vm,vy,clean_val)=source_views()
    x,labels=_classical_training_arrays(seed,tr,tm,y,clean)
    val=noise_source_val(source_strong_root,vr,vm)
    features={"clean":engineered_features(clean_val)}
    features.update({str(k):engineered_features(raw) for k,raw in val.items()})
    cand,artifacts=[],[]
    for index,(params,estimator) in enumerate(classical_candidates(method,seed),1):
        estimator.fit(x,labels)
        clean_f1=metrics(vy,estimator.predict(features["clean"]))["macro_f1"]
        noise_f1={k:metrics(vy,estimator.predict(features[str(k)]))["macro_f1"]
                  for k in config()["validation_noise_levels_db"]}
        row=dict(candidate_index=index,params=params,clean_f1=float(clean_f1),
                 noisy_f1={str(k):float(v) for k,v in noise_f1.items()},
                 noise_mean_f1=float(np.mean(list(noise_f1.values()))))
        cand.append(row);artifacts.append(estimator)
        print(method,seed,params,"source_clean=",round(clean_f1,4),
              "source_noise=",round(row["noise_mean_f1"],4),flush=True)
    best=_selection(cand)
    estimator=artifacts[best["candidate_index"]-1]
    dst.mkdir(parents=True,exist_ok=False)
    joblib.dump(estimator,dst/"best.joblib")
    _write(dst/"run.json",dict(method=method,seed=seed,
        protocol=config()["protocol"],feature_version=FEATURE_VERSION,
        candidates=cand,selected=best,n_source_train=len(y),
        train_augmented_copies=1+config()["classical"]["augmented_source_epochs"],
        train_sample_views=len(labels),source_only=True,
        target_used_for_training_or_selection=False,
        final_holdout_evaluated=False))


def _tensor_predict(model,raw,device,batch=128):
    model.eval(); ys=[]
    with torch.no_grad():
        for i in range(0,len(raw),batch):
            x=torch.from_numpy(raw[i:i+batch]).to(device)
            ys.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(ys)


def _resnet_loss(model,clean,noisy,labels):
    logits=model(torch.cat((clean,noisy),0));n=len(labels)
    return .5*(F.cross_entropy(logits[:n],labels)+F.cross_entropy(logits[n:],labels))


def train_resnet(seed:int,out:Path,source_strong_root:Path):
    method="resnet1d"
    dst=out/"models"/method/f"seed{seed}"
    if dst.exists():raise FileExistsError(f"Refusing overwrite {dst}")
    fix_randomness(seed)
    (tr,tm,y,clean),(vr,vm,vy,clean_val)=source_views()
    val=noise_source_val(source_strong_root,vr,vm)
    v=config()["resnet1d"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FivePhaseResNet1D().to(device)
    optim=torch.optim.Adam(model.parameters(),lr=v["learning_rate"],weight_decay=0.)
    init_hash=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                         for p in model.parameters())).hexdigest()
    history=[]; states={}
    for epoch in range(1,v["epochs"]+1):
        noisy=data.fresh_noisy_source(tr,tm,seed,epoch)
        perm=np.random.default_rng(seed+901+epoch).permutation(len(y))
        model.train(); losses=[]
        for k in range(0,len(perm),v["batch_size"]):
            ids=perm[k:k+v["batch_size"]]
            a=torch.from_numpy(clean[ids]).to(device)
            b=torch.from_numpy(noisy[ids]).to(device)
            yy=torch.from_numpy(y[ids]).to(device)
            optim.zero_grad(set_to_none=True)
            loss=_resnet_loss(model,a,b,yy)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"ResNet loss nonfinite {seed}/{epoch}")
            loss.backward();optim.step()
            losses.append(float(loss.detach().cpu()))
        clean_f1=metrics(vy,_tensor_predict(model,clean_val,device))["macro_f1"]
        noisef={str(k):metrics(vy,_tensor_predict(model,z,device))["macro_f1"] for k,z in val.items()}
        row=dict(epoch=epoch,train_loss_mean=float(np.mean(losses)),
                 clean_f1=float(clean_f1),noisy_f1=noisef,
                 noise_mean_f1=float(np.mean(list(noisef.values()))))
        history.append(row)
        states[epoch]={k:p.detach().cpu().clone() for k,p in model.state_dict().items()}
        if epoch==1 or epoch%10==0 or epoch==v["epochs"]:
            print(f"resnet1d seed{seed} epoch{epoch}/{v['epochs']} "
                  f"srcClean={clean_f1:.4f} src5SNR={row['noise_mean_f1']:.4f}",flush=True)
    best=score_checkpoints(history)
    dst.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state=states[best["epoch"]],method=method,seed=seed,
                    protocol=config()["protocol"],selected=best,source_only=True,
                    final_holdout_evaluated=False),dst/"best.pt")
    _write(dst/"history.json",history)
    _write(dst/"run.json",dict(method=method,seed=seed,protocol=config()["protocol"],
         selected=best,training=v,initial_weights_sha256=init_hash,
         parameters=sum(p.numel() for p in model.parameters()),
         source_only=True,target_used_for_training_or_selection=False,
         final_holdout_evaluated=False))


def train(method,seed,out,source_strong_root):
    if method not in METHODS:raise ValueError(method)
    if seed not in config()["seeds"]:raise ValueError("Seed outside frozen 42/43/44")
    if method=="resnet1d":train_resnet(seed,out,source_strong_root)
    else:train_classical(method,seed,out,source_strong_root)


def _write(path:Path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f"Refusing overwrite {path}")
    path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")


def target_raw(snr,*,snr_root:Path|None,strong_root:Path|None,dev_export_root:Path|None):
    if dev_export_root is not None:
        return data.target_dev_from_export(snr,dev_export_root)
    if snr in (0,5):
        if strong_root is None:raise ValueError("0/5 dB target root required")
        return data.target_dev_from_mat(snr,strong_root/"generated/target")
    return data.target_dev_from_mat(snr,snr_root)


def evaluate(method:str,seed:int,out:Path,conditions:list,*,snr_root=None,
             strong_root=None,dev_export_root=None):
    import joblib
    if method not in METHODS or seed not in config()["seeds"]:raise ValueError("Invalid method/seed")
    folder=out/"models"/method/f"seed{seed}"
    run=json.loads((folder/"run.json").read_text())
    if (run["method"],run["seed"],run["protocol"])!=(method,seed,config()["protocol"]):
        raise ValueError("Incorrect model provenance")
    if run["target_used_for_training_or_selection"] or run["final_holdout_evaluated"]:
        raise ValueError("Training or selection used target data")
    if method=="resnet1d":
        save=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
        if save["method"]!=method or save["seed"]!=seed or save["protocol"]!=config()["protocol"]:
            raise ValueError("Checkpoint provenance mismatch")
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=FivePhaseResNet1D().to(device);model.load_state_dict(save["state"],strict=True)
        predict=lambda x:_tensor_predict(model,x,device)
    else:
        # Only open artifact created by our own train_classical; joblib is unsafe
        # for untrusted external files. Never load user-provided checkpoints.
        estimator=joblib.load(folder/"best.joblib")
        predict=lambda x:estimator.predict(engineered_features(x))
    from prepare import OUT as SOURCE_OUT,read
    frozen=read(SOURCE_OUT/"target_dev_clean_windows.json")
    truth=np.asarray([row["label"] for row in frozen],dtype=np.int64)
    if len(truth)!=2474 or len({row["record_id"] for row in frozen})!=88:
        raise ValueError("Target-Dev must be frozen 88 records/2474 cycles")
    for snr in conditions:
        name="clean" if snr is None else f"snr{snr}"
        dst=out/"evaluations"/method/f"seed{seed}"/f"target_dev_{name}.json"
        if dst.exists():raise FileExistsError(f"Refuse overwrite: {dst}")
        raw,metadata=target_raw(snr,snr_root=snr_root,strong_root=strong_root,
                                dev_export_root=dev_export_root)
        if metadata!=frozen:
            # Some external exports copy the same frozen metadata verbatim;
            # reject any changed cycle identity, boundary, or label.
            raise ValueError(f"Target-Dev metadata changed for {name}")
        x=data.raw_cycles(raw,metadata)
        pred=np.asarray(predict(x),dtype=np.int64)
        score=metrics(truth,pred)
        _write(dst,dict(method=method,seed=seed,protocol=config()["protocol"],condition=name,
             record_families=88,windows=2474,truth=truth.tolist(),predicted=pred.tolist(),
             selected_source=run["selected"],metrics=score,
             synthetic_target_noise=snr in (0,5),
             source_only_selection=True,final_holdout_evaluated=False))
        print("Target-Dev",method,seed,name,"Macro-F1",round(score["macro_f1"],5),flush=True)


def summarize(out:Path,methods, seeds,conditions):
    results=[]; classes=[]; reference_truth=None
    for method in methods:
        for seed in seeds:
            for snr in conditions:
                name="clean" if snr is None else f"snr{snr}"
                record=json.loads((out/"evaluations"/method/f"seed{seed}"/
                                   f"target_dev_{name}.json").read_text())
                expected=(method,seed,config()["protocol"],name)
                got=(record["method"],record["seed"],record["protocol"],record["condition"])
                if (got!=expected or record["windows"]!=2474 or record["record_families"]!=88
                   or record["final_holdout_evaluated"] is not False):
                    raise ValueError("Invalid evaluation provenance")
                if reference_truth is None: reference_truth=record["truth"]
                if record["truth"]!=reference_truth:raise ValueError("Truth/order differs between methods")
                z=record["metrics"]
                results.append(dict(method=method,seed=seed,condition=name,
                     macro_f1_percent=100*z["macro_f1"],accuracy_percent=100*z["accuracy"],
                     fault_as_healthy=z["missed_fault_as_normal"],
                     healthy_as_fault=z["normal_false_alarms"]))
                for cls,(f1,recall) in enumerate(zip(z["per_class_f1"],z["per_class_recall"])):
                    classes.append(dict(method=method,seed=seed,condition=name,label=cls,
                                        f1_percent=100*f1,recall_percent=100*recall))
    summary=[]
    for method in methods:
        for snr in conditions:
            name="clean" if snr is None else f"snr{snr}"
            vals=[row for row in results if row["method"]==method and row["condition"]==name]
            f1=np.asarray([x["macro_f1_percent"] for x in vals]);acc=np.asarray([x["accuracy_percent"] for x in vals])
            summary.append(dict(method=method,condition=name,n_seeds=len(vals),
                macro_f1_mean=float(f1.mean()),macro_f1_sd=float(f1.std(ddof=1)) if len(vals)>1 else 0.,
                accuracy_mean=float(acc.mean()),accuracy_sd=float(acc.std(ddof=1)) if len(vals)>1 else 0.))
    rep=out/"reports"
    rep.mkdir(parents=True,exist_ok=True)
    _csv(rep/"PER_SEED_RESULTS.csv",results)
    _csv(rep/"PER_CLASS_PER_SEED.csv",classes)
    _write(rep/"SUMMARY.json",dict(protocol=config()["protocol"],methods=methods,seeds=seeds,
         target_dev_records=88,final_holdout_evaluated=False,summary=summary))
    print("Reports saved:",rep,flush=True)


def _csv(path,rows):
    if path.exists():raise FileExistsError(path)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def smoke():
    from classical_baseline_models import smoke as model_smoke
    model_smoke()
    v=config()
    assert v["seeds"]==[42,43,44]
    a=[dict(candidate_index=i,clean_f1=c,noise_mean_f1=n)
       for i,(c,n) in enumerate(((1.,.60),(.98,.95),(.995,.87)),start=1)]
    assert _selection(a)["candidate_index"]==3  # .98 is >1pp below clean peak
    rng=np.random.default_rng(42)
    from sklearn.pipeline import Pipeline
    for mode in METHODS[:2]:
        x=rng.normal(size=(36,120));y=np.arange(36)%6
        for params,est in classical_candidates(mode,42):
            est.fit(x,y)
            assert est.predict(x[:3]).shape==(3,)
            break
    m=FivePhaseResNet1D()
    y=torch.tensor([0,1,2])
    x=torch.randn(3,5,4096)
    loss=_resnet_loss(m,x,x+.03*torch.randn_like(x),y)
    loss.backward()
    assert torch.isfinite(loss) and m.classifier.weight.grad is not None
    print("Baseline protocol smoke passed")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("smoke","train","evaluate","summarize"))
    p.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    p.add_argument("--seeds",nargs="+",type=int,default=config()["seeds"])
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--source-strong-root",type=Path)
    p.add_argument("--snr-root",type=Path)
    p.add_argument("--strong-root",type=Path)
    p.add_argument("--dev-export-root",type=Path)
    p.add_argument("--conditions",nargs="+",default=[str(i) for i in DEFAULT_CONDITIONS])
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or any(s not in config()["seeds"] for s in a.seeds):
        p.error("Seeds must be a subset of 42,43,44 without repetition")
    conditions=[None if x=="clean" else int(x) for x in a.conditions]
    if len(set(a.methods))!=len(a.methods) or any(c is not None and c not in DEFAULT_CONDITIONS for c in conditions):
        p.error("Invalid duplicated method or SNR")
    if a.stage=="smoke":smoke()
    elif a.stage=="train":
        if a.source_strong_root is None:p.error("Training requires --source-strong-root")
        for method in a.methods:
            for seed in a.seeds:train(method,seed,a.out,a.source_strong_root)
    elif a.stage=="evaluate":
        if a.snr_root is None and a.dev_export_root is None:p.error("Provide target --snr-root or --dev-export-root")
        if any(c in (0,5) for c in conditions) and a.strong_root is None and a.dev_export_root is None:
            p.error("Strong synthetic 0/5 dB require --strong-root or exported Dev")
        for method in a.methods:
            for seed in a.seeds:
                evaluate(method,seed,a.out,conditions,snr_root=a.snr_root,
                         strong_root=a.strong_root,dev_export_root=a.dev_export_root)
    else:summarize(a.out,a.methods,a.seeds,conditions)

if __name__=="__main__":main()
