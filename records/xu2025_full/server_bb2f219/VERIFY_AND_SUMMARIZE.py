"""独立核验Xu-Full源选模、18组源验证、30组目标预测与同种子历史比较。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/matched"
sys.path.insert(0,str(ROOT/"code/scripts"))
import numpy as np
import torch
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
import xu2025_full_experiment as experiment
import v2_experiment as data
from xu2025_full_model import parameter_counts
from v1_noise_consistency import fix_randomness
from evaluate_v4_strong_snr import frozen_ids,csv_write,CLASS_NAMES


def read(path):
    """读取未舍入的原始JSON，不修改历史结果或预测。"""
    return json.loads(Path(path).read_text())


def write(path,value):
    """验证与比较表仅写本轮目录，保留全部种子。"""
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def sha(path):
    """核对原代码、输入、旧模型和本轮权重的字节。"""
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()


def close(actual,expected):
    """预测向量必须完全相同；独立指标只允许末位舍入。"""
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def statistics(values):
    """仅42/43/44三个匹配种子，样本标准差ddof=1。"""
    assert len(values)==3
    return {"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"values":values}


def independent_metrics(truth,predicted):
    """用sklearn重算六类指标；矩阵行真实列预测。"""
    cm=confusion_matrix(truth,predicted,labels=range(6))
    _,recall,f1,_=precision_recall_fscore_support(truth,predicted,labels=range(6),zero_division=0)
    alarms=int(cm[0,1:].sum())
    return {"macro_f1":float(f1.mean()),"accuracy":float(accuracy_score(truth,predicted)),
            "balanced_accuracy":float(recall.mean()),"per_class_f1":f1.tolist(),"per_class_recall":recall.tolist(),
            "confusion_matrix":cm.tolist(),"normal_false_alarms":alarms,
            "normal_false_alarm_rate":alarms/int(cm[0].sum()),"missed_fault_as_normal":int(cm[1:,0].sum())}


def main():
    """先验证本轮完整模型，再比较原三种子历史分数，不混入六种子均值。"""
    start=time.monotonic()
    assert read(ROOT/"RUN_STATUS.json")["status"]=="completed" and not (OUT/"reports/VERIFICATION.json").exists()
    deployment,preflight=read(ROOT/"DEPLOYMENT_MANIFEST.json"),read(ROOT/"PREFLIGHT.json")
    protected=read(ROOT/"ORIGINAL_FILES_SHA256.json")
    assert all(sha(path)==expected for path,expected in protected.items())
    assert all(sha(ROOT/"code"/path)==expected for path,expected in deployment["code_sha256"].items())
    models,chosen,weights,selections,loss_rows={},{},{},[],[]
    for seed in (42,43,44):
        folder=OUT/f"models/xu_full/seed{seed}"
        history,run=read(folder/"history.json"),read(folder/"run.json")
        assert len(history)==50 and [r["epoch"] for r in history]==list(range(1,51))
        expected={"mode":"xu_full","seed":seed,"track":"matched","epochs":50,"batch":32,"lr":.001,
                  "weight_decay":0.,"step_lr":None,"gamma":None,"source_val_snrs":[0,5,10,20,30],
                  "training_noisy_pair":True,"source_only":True,"target_used_for_training_or_selection":False,
                  "final_holdout_evaluated":False,"loss_config":{"reconstruction":"paper_sum","w_recon":1.,"w_dml":1.}}
        assert all(run[k]==v for k,v in expected.items())
        assert run["initial_model_sha256"]==preflight["initial_model_sha256"][str(seed)]
        for row in history:
            assert set(row["noisy_f1"])=={"0","5","10","20","30"} and row["lr"]==.001
            assert all(np.isfinite(list(row["train_loss"].values())))
            close(row["noise_mean_f1"],np.mean(list(row["noisy_f1"].values())))
            # 各分量由float32算子和每批加权累积，总项核对保留float32舍入容差。
            terms=row["train_loss"]
            np.testing.assert_allclose(terms["total"],terms["ce"]+terms["recon"]+terms["dml"],atol=1e-4,rtol=1e-6)
            loss_rows.append({"seed":seed,"epoch":row["epoch"],**terms,"clean_val_f1_percent":100*row["clean_f1"],
                              "five_snr_val_f1_percent":100*row["noise_mean_f1"]})
        max_clean=max(r["clean_f1"] for r in history)
        eligible=[r for r in history if r["clean_f1"]>=max_clean-.01-1e-12]
        best=sorted(eligible,key=lambda r:(-r["noise_mean_f1"],-r["clean_f1"],r["epoch"]))[0]
        assert best==run["selected"]
        checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
        assert checkpoint["mode"]=="xu_full" and checkpoint["seed"]==seed and checkpoint["track"]=="matched"
        assert checkpoint["selected"]==best and checkpoint["loss_config"]==expected["loss_config"]
        assert checkpoint["source_only"] and not checkpoint["final_holdout_evaluated"]
        for name,value in checkpoint["state"].items():
            if name.endswith("num_batches_tracked"):assert int(value)==best["epoch"]*24
        fix_randomness(seed)
        model=experiment.make_model("xu_full","matched").cuda().eval()
        model.load_state_dict(checkpoint["state"],strict=True)
        assert parameter_counts(model)=={"paper_cae_dml":35475,"spectral":0,"total":35475}
        models[seed],chosen[seed],weights[str(seed)]=model,best,sha(folder/"best.pt")
        selections.append({"seed":seed,"selected_epoch":best["epoch"],"eligible_epochs":len(eligible),
                           "clean_f1_percent":100*best["clean_f1"],"five_snr_mean_percent":100*best["noise_mean_f1"]})
    csv_write(OUT/"reports/TRAIN_LOSS_COMPONENTS.csv",loss_rows)
    csv_write(OUT/"reports/SOURCE_SELECTION.csv",selections)
    vr,vm,vy=data.source_data("val")
    val={"clean":data.raw_cycles(vr,vm)}
    val.update({f"snr{k}":x for k,x in data.fixed_noisy_validation(vr,vm,noise_levels=(0,5,10,20,30),
                                     materialized_root=Path(deployment["source_strong_root"])).items()})
    source_replay=[]
    for seed,model in models.items():
        for condition,x in val.items():
            pred=experiment.predict(model,x,torch.device("cuda"),track="matched")
            m=independent_metrics(vy,pred)
            expected=chosen[seed]["clean_f1"] if condition=="clean" else chosen[seed]["noisy_f1"][condition[3:]]
            close(m["macro_f1"],expected)
            source_replay.append({"seed":seed,"condition":condition,"metrics":m})
    # 3种子×6条件=18组，源验证为当前Xu-Full独立权重的重放。
    assert len(source_replay)==18
    write(OUT/"reports/SOURCE_VAL_REPLAY.json",source_replay)
    print("XU_SOURCE_VAL_REPLAY_PASSED 18 sets",flush=True)
    from prepare import OUT as DATA_OUT,ROOT as DATA_ROOT
    meta=read(DATA_OUT/"target_dev_clean_windows.json")
    frozen_ids(read(DATA_ROOT/"configs/splits_gate0_v1.json"),meta)
    truth=np.array([r["label"] for r in meta])
    cached,classes,per_seed={},[],[]
    conditions=["clean","snr10","snr15","snr20","snr25","snr30","snr35","snr40","snr5","snr0"]
    for condition in conditions:
        snr=None if condition=="clean" else int(condition[3:])
        root=Path(deployment["target_strong_root"]) if snr in (0,5) else ROOT/"target_snr"
        raw,actual=data.target_dev_from_mat(snr,root)
        assert actual==meta
        x=data.raw_cycles(raw,meta)
        for seed,model in models.items():
            result=read(OUT/f"evaluations/xu_full/seed{seed}/target_dev_{condition}.json")
            assert result["selected_source_epoch"]==chosen[seed]["epoch"] and result["track"]=="matched"
            assert result["seed"]==seed and result["mode"]=="xu_full" and result["condition"]==condition
            assert result["record_families"]==88 and result["windows"]==2474 and not result["final_holdout_evaluated"]
            assert not result["target_used_for_training_or_selection"]
            pred=experiment.predict(model,x,torch.device("cuda"),track="matched")
            assert np.array_equal(result["truth"],truth) and np.array_equal(result["predicted"],pred)
            m=independent_metrics(truth,pred)
            for k,v in m.items():close(v,result["metrics"][k])
            cached[seed,condition]=m
            per_seed.append({"seed":seed,"condition":condition,"macro_f1_percent":100*m["macro_f1"],
                             "accuracy_percent":100*m["accuracy"],"fault_as_healthy":m["missed_fault_as_normal"],
                             "healthy_as_fault":m["normal_false_alarms"]})
            for k,name in enumerate(CLASS_NAMES):
                classes.append({"seed":seed,"condition":condition,"label":k,"class_name":name,
                                "f1_percent":100*m["per_class_f1"][k],"recall_percent":100*m["per_class_recall"][k],
                                "fault_as_healthy":m["confusion_matrix"][k][0] if k else ""})
        print("XU_TARGET_REPLAY_PASSED",condition,"3 predictions",flush=True)
    native=read(OUT/"reports/SUMMARY.json")
    aggregates=[]
    for row in native["aggregates"]:
        values=[100*cached[s,row["condition"]]["macro_f1"] for s in (42,43,44)]
        close(row["values"],values);close(row["macro_f1_mean"],np.mean(values));close(row["macro_f1_sample_sd"],np.std(values,ddof=1))
        aggregates.append({"condition":row["condition"],**{k+"_percent":statistics([100*cached[s,row["condition"]][k] for s in (42,43,44)])
                           for k in ("macro_f1","accuracy","balanced_accuracy","normal_false_alarm_rate")}})
    csv_write(OUT/"reports/FULL_PER_SEED_RESULTS.csv",per_seed)
    csv_write(OUT/"reports/PER_CLASS_PER_SEED.csv",classes)
    # 历史V5及三个固定对照均采用42/43/44，不把六种子均值混入配对。
    four=Path(deployment["reference_four_results"])
    pairs,paired_summary,hashes=[],[],{}
    for method in ("identity","lowpass3","lowpass4","nahsf"):
        for condition in conditions:
            deltas=[]
            for seed in (42,43,44):
                path=four/f"evaluations/{method}/seed{seed}/target_dev_{condition}.json"
                old=read(path)
                assert old["seed"]==seed and old["condition"]==condition and old["truth"]==truth.tolist()
                assert old["protocol"]=="four_models_train10_40_val0_5_10_20_30_v1"
                delta=100*(cached[seed,condition]["macro_f1"]-old["metrics"]["macro_f1"])
                deltas.append(delta);hashes[str(path)]=sha(path)
                pairs.append({"reference":method,"seed":seed,"condition":condition,"xu_minus_reference_macro_f1_pp":delta})
            paired_summary.append({"reference":method,"condition":condition,"xu_minus_reference_macro_f1_pp":statistics(deltas)})
    csv_write(OUT/"reports/HISTORICAL_PAIRED_COMPARISON.csv",pairs)
    write(OUT/"reports/VERIFIED_SUMMARY.json",{"track":"matched","mode":"xu_full","seeds":[42,43,44],"std_ddof":1,
         "aggregates":aggregates,"historical_pairs":paired_summary,"historical_results_sha256":hashes})
    assert all(sha(path)==expected for path,expected in protected.items())
    write(OUT/"reports/VERIFICATION.json",{"status":"passed","completed_training_runs":3,"training_epochs":150,
        "source_only_selection_independently_verified":True,"source_validation_metric_sets_replayed":18,
        "target_predictions_replayed":30,"per_class_rows":180,"historical_paired_rows":120,"weights_sha256":weights,
        "protected_original_files_checked":len(protected),"original_files_unchanged":True,"code_unchanged":True,
        "final_holdout_evaluated":False,"source_test_evaluated":False,"elapsed_seconds":time.monotonic()-start})
    print("XU_FULL_INDEPENDENT_VERIFICATION_PASSED",flush=True)


if __name__=="__main__":main()
