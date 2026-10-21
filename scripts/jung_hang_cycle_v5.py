"""Jung/Hang V5: configurable paper-informed Jung/Hang adaptations under the frozen-cycle protocol.

Historical V2/V3 algorithms and measurements remain immutable. This V5 runner
uses one frozen source split, configurable input variants and an explicit
source-only checkpoint rule; no target labels are used for fitting or selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CONFIG=Path(__file__).resolve().parents[1]/"configs/jung_hang_v5_v1.json"


def read(path):return json.loads(Path(path).read_text(encoding="utf8"))


def write(path,value):
    """仅写本轮输出；明确LF便于跨平台逐字节检查。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8",newline="\n")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(args):
    """先导入原数据工具，再导入新模型；公共预处理文件须逐字节相同。"""
    c=read(args.config)
    if not c["models"] or not set(c["models"]).issubset({"jung","hang"}) or len(set(c["models"]))!=len(c["models"]) or c["seeds"]!=[42,44]:
        raise ValueError("Models must be a nonempty Jung/Hang subset and seeds must be 42/44")
    local=Path(__file__).resolve().parent
    reference=args.reference_root/"code/scripts"
    if sha(local/"prepare.py")!=sha(reference/"prepare.py"):raise ValueError("新旧原始输入预处理不同，需先审查")
    sys.path.insert(0,str(reference))
    import source_cycle_benchmark as ref
    import source_cycle_split as split
    sys.path.insert(0,str(local))
    import compare_jung_hang_v5 as paper
    provenance=split.source_provenance(args.source_root)
    if provenance["manifest_sha256"]!=c["source_split_manifest_sha256"]:raise ValueError("源划分改变")
    return c,ref,paper,split,provenance


def noise_records(ref, raw, seed, epoch, *, method, meta, config, snr=None):
    """Jung: paper THD+N schedule; Hang: paper's 15/20 dB Gaussian noise."""
    import numpy as np
    from jung_hang_models_v5 import jung_thdn_augment
    from jung_hang_models_v4 import add_noise as snr_noise
    by_record = {}
    for row in meta:
        by_record.setdefault(row["record_id"], []).append(row)
    output = {}
    for i, rid in enumerate(sorted(raw)):
        time, currents, omega = raw[rid]
        if snr is None:
            rng = np.random.default_rng(seed + 15000 + epoch + 997 * (i + 1))
            if method == "jung":
                levels = config["jung_train_thdn_percent"]
            else:
                levels = config["hang_train_snr_db"]
            level = levels[(epoch + i + seed) % len(levels)]
        else:
            digest = hashlib.sha256(f"HUST-V5:{method}:val:{snr}:{rid}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            level = snr
        if method == "jung" and snr is None:
            noisy = jung_thdn_augment(
                time, currents, omega, level, rng,
                metadata=by_record.get(rid, ()),
                limiter=config["jung_limiter"], quiet_ratio=config["jung_quiet_ratio"],
            )
        else:
            noisy = snr_noise(time, currents, omega, level, rng, mode="awgn")
        output[rid] = (time, noisy, omega)
    return output


def source_data(args,method):
    """Build V5 inputs and labels using source-split-local anchors."""
    import numpy as np
    c,ref,paper,split,provenance=setup(args)
    (raw,tm,y,clean4096),(_,vm,vy,vc4096,_legacy_vnoise4096)=split.source_views(args.source_root)
    if (len(y),len(vy))!=(962,221):raise ValueError("样本数改变")
    clean, train_anchors = paper.build_inputs(method, tm, raw, c)
    y = y[train_anchors]
    val_clean, valid_anchors = paper.build_inputs(method, vm, raw, c)
    vy = vy[valid_anchors]
    val = {"clean": val_clean}
    for snr in c["val_noise_db"]:
        modified = noise_records(ref, raw, 0, 0, method=method, meta=vm, config=c, snr=snr)
        val_input, check_anchors = paper.build_inputs(method, vm, modified, c)
        np.testing.assert_array_equal(check_anchors, valid_anchors)
        val[f"snr{snr}"] = val_input
    return c, ref, paper, provenance, raw, tm, y, clean, vy, val


def predict(ref,model,x,y=None):
    """直接复用基准的按样本加权验证CE及推理批大小。"""
    import torch
    return ref.predict(model,x,torch.device("cuda"),y)


def train(args):
    """Train V5 paper-oriented adaptations on simulation source only."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    c,ref,paper,provenance,raw,tm,y,clean,vy,val=source_data(args,args.method)
    method,seed=args.method,args.seed
    if method not in c["models"] or seed not in c["seeds"]:
        raise ValueError("Method/seed does not match experiment config")
    folder=args.output/f"models/{method}/seed{seed}"
    if folder.exists():raise FileExistsError(folder)
    ref.fix_randomness(seed)
    model=paper.get_model(method,c).cuda()
    from jung_hang_optim_v5 import AdaMod
    optimizer=(AdaMod(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"]) if method=="jung" else torch.optim.Adam(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"]))
    initial=ref.digest_parameters(model)
    states,history={},[]
    for epoch in range(1,c["epochs_by_method"][method]+1):
        input_hash=None
        noisy=None
        if method=="jung" or (method=="hang" and c["hang_training_view"]=="paired"):
            modified=noise_records(ref,raw,seed,epoch,method=method,meta=tm,config=c)
            input_hash=hashlib.sha256(ref.data.raw_cycles(modified,tm).tobytes()).hexdigest()
            noisy, noisy_anchor=paper.build_inputs(method,tm,modified,c)
            if len(noisy)!=len(y):raise ValueError("input and label counts differ")
        order=np.random.default_rng(seed+901+epoch).permutation(len(y))
        model.train();losses=[];weighted=0.0
        for start in range(0,len(order),c["batch_size"]):
            ids=order[start:start+c["batch_size"]]
            x_clean=torch.from_numpy(clean[ids]).cuda()
            labels=torch.from_numpy(y[ids]).cuda()
            optimizer.zero_grad(set_to_none=True)
            if method=="jung" or (method=="hang" and c["hang_training_view"]=="paired"):
                x_noisy=torch.from_numpy(noisy[ids]).cuda()
                n=len(ids)
                scores=model(torch.cat((x_clean,x_noisy),dim=0))
                loss=0.5*(F.cross_entropy(scores[:n],labels)+F.cross_entropy(scores[n:],labels))
            else:
                scores=model(x_clean)
                loss=F.cross_entropy(scores,labels)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("non-finite loss")
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("non-finite gradient")
            optimizer.step();scalar=float(loss.detach().cpu());losses.append(scalar);weighted+=scalar*len(ids)
        scores={};ces={}
        for name,values in val.items():
            pred,ce=predict(ref,model,values,vy)
            scores[name]=ref.metrics(vy,pred)["macro_f1"];ces[name]=ce
        row={"epoch":epoch,"train_loss_mean":float(np.mean(losses)),
            "train_loss_sample_weighted":weighted/len(y),"optimizer_steps":len(losses),
            "clean_f1":scores["clean"],"clean_ce":ces["clean"],
            "noisy_f1":{str(s):scores[f"snr{s}"] for s in c["val_noise_db"]},
            "noisy_ce":{str(s):ces[f"snr{s}"] for s in c["val_noise_db"]},
            "noise_mean_f1":float(np.mean([scores[f"snr{s}"] for s in c["val_noise_db"]])),
            "noise_mean_ce":float(np.mean([ces[f"snr{s}"] for s in c["val_noise_db"]])),
            "raw_training_noise_sha256":input_hash,
            "sampling_order_sha256":hashlib.sha256(order.tobytes()).hexdigest()}
        history.append(row);states[epoch]=ref.snapshot(model)
        if epoch==1 or epoch%10==0:
            print(f"{method} seed{seed} epoch{epoch}/{c['epochs_by_method'][method]} loss={row['train_loss_mean']:.6f} sourceF1={row['noise_mean_f1']:.6f}",flush=True)
    chosen=ref.select_source_checkpoint(history)
    folder.mkdir(parents=True)
    common={"method":method,"seed":seed,"protocol":c["protocol"],"config":c,
        "source_split":provenance,"selected":chosen,"target_used_for_training_or_selection":False,
        "final_holdout_evaluated":False}
    torch.save({**common,"state":states[chosen["epoch"]]},folder/"best.pt")
    torch.save({**common,"state":states[c["epochs_by_method"][method]],"epoch":c["epochs_by_method"][method]},folder/"last.pt")
    write(folder/"history.json",history)
    write(folder/"run.json",{**common,"parameters":sum(p.numel() for p in model.parameters()),
        "initial_parameters_sha256":initial,"source_train_cycles":len(y),"source_val_cycles":len(vy),
        "training_view":("paired_"+("thdn" if method=="jung" else "awgn") if method=="jung" or c["hang_training_view"]=="paired" else "clean_only"),
        "first_training_noise_sha256":history[0]["raw_training_noise_sha256"]})


def target_root(args,snr):
    """复用已存在七档数据；不生成任何目标噪声，不读取最终保留集。"""
    m=read(args.reference_root/"DEPLOYMENT_MANIFEST.json")
    if snr in (4,-4,-8):return args.additional_root/"data"
    return Path(m["strong_root"] if snr in (0,5) else m["extreme_root"])


def evaluate(args,replay=False):
    """同一档位每个模型只制作一次输入；两个种子共同使用相同输入数组。"""
    import numpy as np
    import torch
    c,ref,paper,split,provenance=setup(args)
    from sklearn.metrics import f1_score,accuracy_score,recall_score,confusion_matrix
    models={}
    for method in c["models"]:
        for seed in c["seeds"]:
            saved=torch.load(args.output/f"models/{method}/seed{seed}/best.pt",map_location="cpu",weights_only=True)
            if saved["config"]!=c or saved["source_split"]!=provenance:raise ValueError("权重版本不符")
            ref.fix_randomness(seed);model=paper.get_model(method,c).cuda().eval();model.load_state_dict(saved["state"],strict=True)
            models[method,seed]=(model,saved)
    states={key:ref.snapshot(pair[0]) for key,pair in models.items()}
    count=0
    for snr in c["target_conditions_db"]:
        raw,metadata=ref.data.target_dev_from_mat(snr,target_root(args,snr))
        if len(raw)!=88 or len(metadata)!=2474:raise ValueError("Dev队列改变")
        truth=np.asarray([r["label"] for r in metadata],dtype=np.int64)
        common=None
        if c.get("target_common_anchors",False):
            from jung_hang_models_v5 import forward_two_cycle_indices
            common=np.asarray([row[0] for row in forward_two_cycle_indices(metadata)],dtype=np.int64)
        for method in c["models"]:
            values,anchors=paper.build_inputs(method,metadata,raw,c)
            if common is not None:
                keep=np.isin(anchors,common)
                values,anchors=values[keep],anchors[keep]
                np.testing.assert_array_equal(anchors,common)
            method_truth=truth[anchors]
            for seed in c["seeds"]:
                model,saved=models[method,seed];pred,_=predict(ref,model,values)
                score=ref.metrics(method_truth,pred)
                path=args.output/f"evaluations/{method}/seed{seed}/target_dev_snr{snr}.json"
                if replay:
                    obj=read(path);np.testing.assert_array_equal(obj["truth"],method_truth);np.testing.assert_array_equal(obj["predicted"],pred)
                    np.testing.assert_allclose(obj["metrics"]["macro_f1"],f1_score(method_truth,pred,average="macro"),rtol=0,atol=1e-15)
                    np.testing.assert_allclose(obj["metrics"]["accuracy"],accuracy_score(method_truth,pred),rtol=0,atol=1e-15)
                    np.testing.assert_allclose(obj["metrics"]["per_class_f1"],f1_score(method_truth,pred,average=None),rtol=0,atol=1e-15)
                    np.testing.assert_allclose(obj["metrics"]["per_class_recall"],recall_score(method_truth,pred,average=None),rtol=0,atol=1e-15)
                    np.testing.assert_array_equal(obj["metrics"]["confusion_matrix"],confusion_matrix(method_truth,pred,labels=list(range(6))))
                else:
                    if path.exists():raise FileExistsError(path)
                    write(path,{"method":method,"seed":seed,"condition":f"snr{snr}","metrics":score,"truth":method_truth.tolist(),"predicted":pred.tolist(),"original_cycle_anchors":anchors.tolist(),
                        "config":c,"source_split":provenance,"record_families":88,"windows":len(anchors),"selected_source_epoch":saved["selected"]["epoch"],
                        "target_used_for_training_or_selection":False,"final_holdout_evaluated":False})
                count+=1
                print(f"{'REPLAY' if replay else 'DEV'} {method} seed{seed} {snr}dB F1={100*score['macro_f1']:.3f}",flush=True)
    for key,(model,_) in models.items():
        if not all(torch.equal(v.detach().cpu(),states[key][k]) for k,v in model.state_dict().items()):raise ValueError("推理改变模型状态")
    if replay:
        # 重放源验证与选模，并核对方法特定训练视图和共同的采样顺序。
        source_rows=[]
        for method in c["models"]:
            _,_,_,_,_,_,_,_,vy,val=source_data(args,method)
            for seed in c["seeds"]:
                model,saved=models[method,seed];history=read(args.output/f"models/{method}/seed{seed}/history.json")
                chosen=ref.select_source_checkpoint(history)
                if chosen!=saved["selected"]:raise ValueError("原best轮次不符")
                for name,x in val.items():
                    pred,ce=predict(ref,model,x,vy)
                    expected_f1=chosen["clean_f1"] if name=="clean" else chosen["noisy_f1"][name[3:]]
                    expected_ce=chosen["clean_ce"] if name=="clean" else chosen["noisy_ce"][name[3:]]
                    np.testing.assert_allclose(ref.metrics(vy,pred)["macro_f1"],expected_f1,rtol=0,atol=1e-15)
                    np.testing.assert_allclose(ce,expected_ce,rtol=0,atol=1e-9)
                    source_rows.append({"method":method,"seed":seed,"condition":name,"macro_f1":expected_f1,"ce":ce})
        for seed in c["seeds"]:
            histories={method:read(args.output/f"models/{method}/seed{seed}/history.json")
                       for method in c["models"]}
            for method, history in histories.items():
                if len(history)!=c["epochs_by_method"][method]:
                    raise ValueError("Training epochs do not match config")
                should_have_noise=method=="jung" or c["hang_training_view"]=="paired"
                if any((row["raw_training_noise_sha256"] is not None)!=should_have_noise for row in history):
                    raise ValueError(f"{method} training-noise provenance mismatch")
            # Jung uses two-cycle anchors (fewer samples) and 30 AdaMod epochs;
            # Hang uses one-cycle anchors and 50 Adam epochs. Equal orders/steps
            # across methods are not an applicable validation invariant.
        ref.write_csv(args.output/"reports/SOURCE_VALIDATION_REPLAY.csv",source_rows)
        write(args.output/"VERIFICATION.json",{"status":"passed","target_prediction_replays":count,"source_validation_replays":len(source_rows),
            "paired_sampling_order_checked":False,"training_noise_by_method":{"jung":"thdn","hang":"awgn"},"hang_training_view":c["hang_training_view"],"model_parameters_and_bn_unchanged":True,"final_holdout_evaluated":False})


def summarize(args):
    """保存两新模型逐种子结果，并与原十二模型七档指标客观合表。"""
    import numpy as np
    c,ref,_,_,_=setup(args)
    if read(args.output/"VERIFICATION.json")["status"]!="passed":raise ValueError("未核验")
    rows=[];classes=[];history_rows=[];selected=[]
    for method in c["models"]:
        for seed in c["seeds"]:
            run=read(args.output/f"models/{method}/seed{seed}/run.json")
            selected.append({"method":method,"seed":seed,"selected_epoch":run["selected"]["epoch"],"parameters":run["parameters"]})
            for row in read(args.output/f"models/{method}/seed{seed}/history.json"):
                history_rows.append({"method":method,"seed":seed,**{k:v for k,v in row.items() if not isinstance(v,dict)}})
            for snr in c["target_conditions_db"]:
                obj=read(args.output/f"evaluations/{method}/seed{seed}/target_dev_snr{snr}.json");s=obj["metrics"]
                rows.append({"method":method,"seed":seed,"snr_db":snr,"macro_f1_percent":100*s["macro_f1"],"accuracy_percent":100*s["accuracy"],
                    "fault_as_healthy":s["missed_fault_as_normal"],"healthy_as_fault":s["normal_false_alarms"]})
                classes.extend({"method":method,"seed":seed,"snr_db":snr,"label":i,"f1_percent":100*f,"recall_percent":100*r}
                    for i,(f,r) in enumerate(zip(s["per_class_f1"],s["per_class_recall"])))
    import csv
    with (args.previous_output/"results/reports/PER_SEED_RESULTS.csv").open(encoding="utf8") as stream:
        previous=list(csv.DictReader(stream))
    if len(previous)!=168:raise ValueError("原十二模型记录不全")
    combined=previous+rows
    aggregate=[]
    for method in dict.fromkeys(r["method"] for r in combined):
        for snr in c["target_conditions_db"]:
            subset=[r for r in combined if r["method"]==method and int(r["snr_db"])==snr]
            values=[float(r["macro_f1_percent"]) for r in subset]
            if len(values)!=2:raise ValueError("种子未配对")
            aggregate.append({"method":method,"snr_db":snr,"macro_f1_mean":float(np.mean(values)),"macro_f1_sd":float(np.std(values,ddof=1))})
    for name,data in (("PER_SEED_RESULTS.csv",rows),("PER_CLASS_PER_SEED.csv",classes),("TRAINING_HISTORY.csv",history_rows),
                      ("SELECTED_CHECKPOINTS.csv",selected),("ALL_14_MODELS_PER_SEED.csv",combined),("ALL_14_MODELS_SUMMARY.csv",aggregate)):
        ref.write_csv(args.output/"reports"/name,data)
    print(f"SUMMARY: {len(rows)} new tests, {len(combined)} combined tests, models={','.join(c['models'])}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("preflight","train","evaluate","verify","summarize"))
    parser.add_argument("--method",choices=("jung","hang"))
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--seed",type=int,choices=(42,44))
    for name in ("reference-root","source-root","additional-root","previous-output","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();start=time.time()
    if args.stage=="preflight":
        conf=read(args.config)
        for method in conf["models"]:
            conf,ref,paper,prov,raw,metadata,y,clean,vy,val=source_data(args,method)
            print(f"READY {method} source_train={len(y)} input_shape={list(clean.shape)} "
                  f"source_val={len(vy)} val_shape={list(val['clean'].shape)}",flush=True)
            if method == "hang":
                from hang_v6_source_audit import compare_hang_normalizers
                contrast = compare_hang_normalizers(metadata, raw, y, conf)
                write(args.output/"PREFLIGHT_HANG_SOURCE_COLLISIONS.json",contrast)
                print("HANG SOURCE-ONLY IMAGE COLLISIONS",contrast,flush=True)
        from jung_hang_v5_audit import audit_source
        audit = audit_source(raw, metadata, conf)
        write(args.output/"PREFLIGHT_PAPER_AUDIT.json",audit)
        print("SOURCE PAPER-ASSUMPTION AUDIT",audit,flush=True)
        print("SMOKE",paper.smoke(conf),flush=True)
    elif args.stage=="train":
        if args.method is None or args.seed is None:parser.error("训练必须指定模型和种子")
        train(args)
    elif args.stage=="evaluate":evaluate(args)
    elif args.stage=="verify":evaluate(args,replay=True)
    else:summarize(args)
    print(f"{args.stage} finished in {time.time()-start:.3f}s",flush=True)


if __name__=="__main__":main()
