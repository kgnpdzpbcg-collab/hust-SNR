"""补齐四个基线并汇总十二模型：只新增LP3的42/44三档共六组推理。

沿用已核验八模型入口的只读加载与数据工具；不修改该入口、模型或训练。
ResNet、Plain CNN、LP4的18份三档预测，以及其他144份记录均原字节引用。
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import shutil
import time

import b1_seven_snr_fixed_weights as common

CONFIG=Path(__file__).resolve().parents[1]/"configs/twelve_model_seven_snr_v1.json"
ADDED=("resnet1d","cnn1d_plain","resnet_lp3","resnet_lp4")
MODELS=common.MODELS+ADDED
read,write,sha,save_csv=common.read,common.write,common.sha,common.save_csv


def config():
    """冻结新增四模型、两种子及七档条件，数据沿用八模型已核验协议。"""
    c=read(CONFIG)
    if c["additional_models"]!=list(ADDED) or c["seeds"]!=list(common.SEEDS) or c["conditions_db"]!=list(common.CONDITIONS):
        raise ValueError("本轮配置改变")
    return c


def prior(args,method,seed,snr):
    """返回已有评价；仅LP3的三个新增档位返回None，明确需要本轮推理。"""
    relative=f"evaluations/{method}/seed{seed}/target_dev_snr{snr}.json"
    if method in common.MODELS:
        return args.previous_output/"results"/relative
    if snr in common.EXTRA:
        return None if method=="resnet_lp3" else args.additional_root/"evaluation"/relative
    return args.baseline_root/"results"/relative


def load(args,original,ablation,provenance):
    """复用原八模型严格加载，再从原基准快照严格加载四个新增基线。"""
    import torch
    from v1_noise_consistency import fix_randomness
    models,runs=common.load_models(args,original,ablation,provenance)
    for method in ADDED:
        for seed in common.SEEDS:
            folder=args.baseline_root/f"results/models/{method}/seed{seed}"
            saved=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
            run=read(folder/"run.json")
            if (saved["method"],saved["seed"],saved["config"],saved["source_split"],saved["selected"])!=(method,seed,original.configuration(),provenance,run["selected"]):
                raise ValueError("新增基线权重来源不符")
            fix_randomness(seed)
            model=original.build_model(method).cuda().eval()
            model.load_state_dict(saved["state"],strict=True)
            models[method,seed],runs[method,seed]=model,run
    return models,runs


def preflight(args):
    """重新核对既有输入指纹、24份最佳权重及162份可复用预测。"""
    config()
    _,original,ablation,provenance,_=common.setup(args)
    if args.output.exists():raise FileExistsError(args.output)
    if read(args.previous_output/"VERIFICATION.json")["status"]!="passed":raise ValueError("八模型结果未核验")
    protected=read(args.previous_output/"PROTECTED_FILES_SHA256.json")
    for path,expected in protected.items():
        if sha(path)!=expected:raise ValueError(f"既有保护清单不符：{path}")
    for method in MODELS:
        for seed in common.SEEDS:
            folder=common.checkpoint_folder(args,method,seed)
            for name in ("best.pt","last.pt","run.json","history.json"):
                path=folder/name;protected[str(path)]=sha(path)
            for snr in common.CONDITIONS:
                path=prior(args,method,seed,snr)
                if path is not None:protected[str(path)]=sha(path)
    for path,expected in protected.items():
        if sha(path)!=expected:raise ValueError(f"既有文件变化：{path}")
    models,_=load(args,original,ablation,provenance)
    count=sum(prior(args,m,s,d) is not None for m in MODELS for s in common.SEEDS for d in common.CONDITIONS)
    if count!=162 or len(models)!=24:raise ValueError("模型或复用数量不符")
    args.output.mkdir(parents=True)
    write(args.output/"PROTECTED_FILES_SHA256.json",protected)
    write(args.output/"PREFLIGHT.json",{"status":"passed","config":config(),"source_split":provenance,
        "models_loaded_strictly":len(models),"reused_prediction_files":count,"protected_files":len(protected)})
    print("PREFLIGHT PASSED: 24 best weights, 162 prior predictions",flush=True)


def evaluate(args):
    """复制旧162份JSON，LP3仅用原best推理三档，独立另存六份新预测。"""
    import numpy as np
    import torch
    _,original,ablation,provenance,metadata=common.setup(args)
    common.assert_protected(args)
    destination=args.output/"results/evaluations"
    if destination.exists():raise FileExistsError(destination)
    sources=[]
    for method in MODELS:
        for seed in common.SEEDS:
            for snr in common.CONDITIONS:
                path=prior(args,method,seed,snr)
                if path is None:continue
                target=destination/f"{method}/seed{seed}/target_dev_snr{snr}.json"
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
                sources.append({"method":method,"seed":seed,"snr_db":snr,"kind":"reused","source_path":str(path),"source_sha256":sha(path)})
    models,runs=load(args,original,ablation,provenance)
    truth=np.asarray([r["label"] for r in metadata],dtype=np.int64)
    for snr in common.EXTRA:
        raw,actual=original.data.target_dev_from_mat(snr,common.condition_root(args,snr))
        if actual!=metadata:raise ValueError("周期顺序改变")
        x=original.data.raw_cycles(raw,metadata)
        for seed in common.SEEDS:
            pred,_=original.predict(models["resnet_lp3",seed],x,torch.device("cuda"))
            score=original.metrics(truth,pred)
            path=destination/f"resnet_lp3/seed{seed}/target_dev_snr{snr}.json"
            write(path,{"protocol":config()["protocol"],"method":"resnet_lp3","seed":seed,"condition":f"snr{snr}",
                "truth":truth.tolist(),"predicted":pred.tolist(),"metrics":score,"source_split":provenance,
                "source_only_selection":runs["resnet_lp3",seed]["selected"],"windows":2474,"record_families":88,
                "target_used_for_training_or_selection":False,"final_holdout_evaluated":False})
            sources.append({"method":"resnet_lp3","seed":seed,"snr_db":snr,"kind":"new","source_path":str(path),"source_sha256":sha(path)})
            print(f"NEW LP3 seed{seed} {snr}dB Macro-F1={100*score['macro_f1']:.4f}",flush=True)
    if len(sources)!=168:raise ValueError("评价未齐全")
    write(args.output/"RESULT_PROVENANCE.json",sources)
    common.assert_protected(args)


def verify(args):
    """回放全部168组目标预测与144组源验证，独立复算六类指标和误报漏报。"""
    import numpy as np
    import torch
    from sklearn.metrics import f1_score,accuracy_score,recall_score,confusion_matrix
    _,original,ablation,provenance,metadata=common.setup(args)
    models,runs=load(args,original,ablation,provenance)
    before={key:original.snapshot(model) for key,model in models.items()}
    (_,val),_=original.checked_source(args.source_root,original.configuration())
    _,_,vy,clean,noisy=val
    source_rows=[]
    for key,model in models.items():
        selected=runs[key]["selected"]
        for name,x in [("clean",clean),*[(f"snr{s}",x) for s,x in noisy.items()]]:
            pred,ce=original.predict(model,x,torch.device("cuda"),vy)
            expected_f1=selected["clean_f1"] if name=="clean" else selected["noisy_f1"][name[3:]]
            expected_ce=selected["clean_ce"] if name=="clean" else selected["noisy_ce"][name[3:]]
            np.testing.assert_allclose(f1_score(vy,pred,average="macro"),expected_f1,rtol=0,atol=1e-15)
            np.testing.assert_allclose(ce,expected_ce,rtol=0,atol=1e-9)
            source_rows.append({"method":key[0],"seed":key[1],"condition":name,"ce":ce,"macro_f1":expected_f1})
    truth=np.asarray([r["label"] for r in metadata],dtype=np.int64)
    for snr in common.CONDITIONS:
        raw,actual=original.data.target_dev_from_mat(snr,common.condition_root(args,snr))
        if actual!=metadata:raise ValueError("目标周期顺序改变")
        x=original.data.raw_cycles(raw,metadata)
        for (method,seed),model in models.items():
            obj=read(args.output/f"results/evaluations/{method}/seed{seed}/target_dev_snr{snr}.json")
            if (obj["method"],obj["seed"],obj["condition"])!=(method,seed,f"snr{snr}"):raise ValueError("预测身份不符")
            pred,_=original.predict(model,x,torch.device("cuda"))
            np.testing.assert_array_equal(obj["truth"],truth);np.testing.assert_array_equal(obj["predicted"],pred)
            score=obj["metrics"];cm=confusion_matrix(truth,pred,labels=list(range(6)))
            np.testing.assert_allclose(score["macro_f1"],f1_score(truth,pred,average="macro"),rtol=0,atol=1e-15)
            np.testing.assert_allclose(score["accuracy"],accuracy_score(truth,pred),rtol=0,atol=1e-15)
            np.testing.assert_allclose(score["per_class_f1"],f1_score(truth,pred,average=None),rtol=0,atol=1e-15)
            np.testing.assert_allclose(score["per_class_recall"],recall_score(truth,pred,average=None),rtol=0,atol=1e-15)
            np.testing.assert_array_equal(score["confusion_matrix"],cm)
            if score["missed_fault_as_normal"]!=int(cm[1:,0].sum()) or score["normal_false_alarms"]!=int(cm[0,1:].sum()):raise ValueError("误报漏报不符")
    for key,model in models.items():
        if not all(torch.equal(v.detach().cpu(),before[key][k]) for k,v in model.state_dict().items()):raise ValueError("推理改变参数或BN")
    for row in read(args.output/"RESULT_PROVENANCE.json"):
        target=args.output/f"results/evaluations/{row['method']}/seed{row['seed']}/target_dev_snr{row['snr_db']}.json"
        if sha(target)!=row["source_sha256"] or sha(Path(row["source_path"]))!=row["source_sha256"]:raise ValueError("来源字节不符")
    protected=common.assert_protected(args)
    reports=args.output/"results/reports";reports.mkdir(parents=True)
    save_csv(reports/"SOURCE_VALIDATION_REPLAY.csv",source_rows)
    write(args.output/"VERIFICATION.json",{"status":"passed","exact_target_prediction_replays":168,"source_validation_replays":144,
        "independent_sklearn_metric_checks":168,"reused_prediction_files_byte_identical":162,"new_evaluations":6,
        "protected_files_unchanged":len(protected),"model_parameters_and_bn_unchanged":True,"retrained":False,"final_holdout_evaluated":False})
    print("VERIFIED: 168 target predictions, 144 source validation conditions",flush=True)


def summarize(args):
    """记录十二模型七档的实际种子、六类指标、均值/样本SD和同种子配对差。"""
    import numpy as np
    if read(args.output/"VERIFICATION.json")["status"]!="passed":raise ValueError("未通过核验")
    rows,classes,summary,pairs=[],[],[],[]
    for method in MODELS:
        for seed in common.SEEDS:
            for snr in common.CONDITIONS:
                obj=read(args.output/f"results/evaluations/{method}/seed{seed}/target_dev_snr{snr}.json");score=obj["metrics"]
                rows.append({"method":method,"seed":seed,"snr_db":snr,"macro_f1_percent":100*score["macro_f1"],
                    "accuracy_percent":100*score["accuracy"],"fault_as_healthy":score["missed_fault_as_normal"],"healthy_as_fault":score["normal_false_alarms"]})
                classes.extend({"method":method,"seed":seed,"snr_db":snr,"label":label,"f1_percent":100*f1,"recall_percent":100*recall}
                    for label,(f1,recall) in enumerate(zip(score["per_class_f1"],score["per_class_recall"])))
        for snr in common.CONDITIONS:
            data=[r for r in rows if r["method"]==method and r["snr_db"]==snr]
            item={"method":method,"snr_db":snr,"seeds":len(data)}
            for metric in ("macro_f1","accuracy"):
                values=[r[metric+"_percent"] for r in data]
                item[metric+"_mean"]=float(np.mean(values));item[metric+"_sd"]=float(np.std(values,ddof=1))
            summary.append(item)
    for row in rows:
        if row["method"]=="b1":continue
        full=next(r for r in rows if r["method"]=="b1" and r["seed"]==row["seed"] and r["snr_db"]==row["snr_db"])
        pairs.append({"method":row["method"],"seed":row["seed"],"snr_db":row["snr_db"],"full_minus_model_pp":full["macro_f1_percent"]-row["macro_f1_percent"]})
    if (len(rows),len(classes),len(summary),len(pairs))!=(168,1008,84,154):raise ValueError("汇总数量不符")
    for name,values in (("PER_SEED_RESULTS.csv",rows),("PER_CLASS_PER_SEED.csv",classes),("SUMMARY.csv",summary),("PAIRED_FULL_B1_MINUS_MODELS.csv",pairs)):
        save_csv(args.output/"results/reports"/name,values)
    write(args.output/"results/reports/SUMMARY.json",{"config":config(),"models":list(MODELS),"summary":summary})
    print("SUMMARY: 168 seed results, 1008 class rows, 84 means, 154 pairs",flush=True)


def main():
    """阶段显式调用；失败保留现场，不自动覆盖或重试。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("preflight","evaluate","verify","summarize"))
    for name in ("baseline-root","ablation-root","additional-root","source-root","previous-output","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();started=time.time()
    {"preflight":preflight,"evaluate":evaluate,"verify":verify,"summarize":summarize}[args.stage](args)
    print(f"{args.stage}: {time.time()-started:.3f}s",flush=True)


if __name__=="__main__":main()
