"""独立核查训练轨迹、五档选模、验证饱和及120组实测预测。"""
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "code/scripts"))
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import v2_experiment as data
from four_models_expanded_val import config, make_model, check_checkpoint, METHODS
from v1_noise_consistency import fix_randomness
from v5_experiment import predict_with_gate, gate_summary


def read(path):
    """核对服务器原文件，始终保留完整数值精度。"""
    return json.loads(Path(path).read_text())


def write(path, value):
    """只写验证结论及指标，不导出原始电流或权重。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    """逐块核对旧文件以及本轮输入、代码和权重。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    """sklearn独立计算指标仅允许末位舍入差。"""
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


def stats(values):
    """同种子配对后再汇总，样本标准差ddof=1。"""
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}


def csv_save(path, rows):
    """用平表保留每个种子的实际观察值。"""
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def independent_metrics(truth, prediction):
    """明确行是真实类别、列是预测类别，分开健康误报与故障漏报。"""
    cm = confusion_matrix(truth, prediction, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=range(6), zero_division=0)
    alarm = int(cm[0, 1:].sum())
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "normal_false_alarms": alarm,
            "normal_false_alarm_rate": alarm / int(cm[0].sum()), "missed_fault_as_normal": int(cm[1:, 0].sum())}


def main():
    """验证只改变选模：旧选模审计快照必须与原最佳权重逐张量完全一致。"""
    start = time.monotonic()
    assert read(ROOT / "RUN_STATUS.json")["status"] == "completed" and not (OUT / "VERIFICATION.json").exists()
    deployment, preflight, protocol = read(ROOT / "DEPLOYMENT_MANIFEST.json"), read(ROOT / "PREFLIGHT.json"), config()
    original = read(ROOT / "ORIGINAL_FILES_SHA256.json")
    for p, expected in original.items():
        assert sha(p) == expected, p
    for p, expected in deployment["code_sha256"].items():
        assert sha(ROOT / "code" / p) == expected, p
    seeds = protocol["seeds"]
    conditions = ["clean" if x == "clean" else f"snr{x}" for x in protocol["target_conditions"]]
    models, checkpoints, weights, selections, saturation, trajectory = {}, {}, {}, {}, [], []
    for method, seed in itertools.product(METHODS, seeds):
        key = f"{method}/seed{seed}"
        folder = OUT / "models" / key
        history, info = read(folder / "history.json"), read(folder / "run.json")
        assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
        assert info["protocol"] == protocol and info["initial_cnn_sha256"] == preflight["initial_cnn_sha256"][str(seed)][method]
        assert not info["target_used_for_training_or_selection"]
        old_folder = Path(deployment["reference_roots"][method]) / "results/models" / key
        old_history = read(old_folder / "history.json")
        assert len(old_history) == 50
        # 不仅训练设置相同，还要求全部50轮loss及原三档验证分数完全复现。
        for row, previous in zip(history, old_history):
            assert row["train_loss_mean"] == previous["train_loss_mean"], (key, row["epoch"], "training loss differs")
            assert row["clean_f1"] == previous["clean_f1"]
            assert set(row["noisy_f1"]) == {"0", "5", "10", "20", "30"}
            assert all(row["noisy_f1"][str(k)] == previous["noisy_f1"][str(k)] for k in (10, 20, 30))
            close(row["noise_mean_f1"], np.mean(list(row["noisy_f1"].values())))
            if method == "nahsf":
                assert row["gate_bias_value"] == previous["gate_bias_value"]
                assert row["gate_bias_grad_abs_mean"] == previous["gate_bias_grad_abs_mean"]
        maximum = max(r["clean_f1"] for r in history)
        eligible = [r for r in history if r["clean_f1"] >= maximum-.01-1e-12]
        chosen = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
        old_choice = sorted(eligible, key=lambda r: (-r["original_three_snr_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
        assert chosen == info["selected"] and old_choice["epoch"] == info["old_three_snr_selected"]["epoch"]
        reference = torch.load(folder / "old_selection_reference.pt", map_location="cpu", weights_only=True)
        old_weight = torch.load(old_folder / "best.pt", map_location="cpu", weights_only=True)
        assert reference["epoch"] == old_choice["epoch"] and reference["state"].keys() == old_weight["state"].keys()
        assert all(torch.equal(value, old_weight["state"][name]) for name, value in reference["state"].items()), key
        trajectory.append({"method": method, "seed": seed, "epochs_training_loss_exactly_equal": 50,
                           "original_clean_three_noise_scores_exactly_equal": True, "old_best_weight_state_exactly_equal": True,
                           "old_selected_epoch": old_choice["epoch"], "new_selected_epoch": chosen["epoch"]})
        saturation.append({"method": method, "seed": seed, "old_selected_epoch": old_choice["epoch"], "new_selected_epoch": chosen["epoch"],
              "original_three_snr_max_percent": 100*max(r["original_three_snr_mean_f1"] for r in history),
              "expanded_five_snr_max_percent": 100*max(r["noise_mean_f1"] for r in history),
              "original_three_snr_epochs_at_100": sum(r["original_three_snr_mean_f1"] >= 1-1e-12 for r in history),
              "expanded_five_snr_epochs_at_100": sum(r["noise_mean_f1"] >= 1-1e-12 for r in history),
              "eligible_epochs": len(eligible), "eligible_distinct_three_scores": len({round(r["original_three_snr_mean_f1"], 12) for r in eligible}),
              "eligible_distinct_five_scores": len({round(r["noise_mean_f1"], 12) for r in eligible})})
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        check_checkpoint(checkpoint, method, seed)
        assert checkpoint["selected"] == chosen
        for name, value in checkpoint["state"].items():
            if name.endswith("num_batches_tracked"):
                assert int(value) == chosen["epoch"]*24
        fix_randomness(seed)
        model = make_model(method).cuda(); model.load_state_dict(checkpoint["state"], strict=True); model.eval()
        models[key], checkpoints[key], selections[key], weights[key] = model, checkpoint, chosen, sha(folder / "best.pt")
    csv_save(OUT / "VALIDATION_SATURATION.csv", saturation)
    write(OUT / "TRAINING_TRAJECTORY_PARITY.json", trajectory)
    print("TRAJECTORY_PARITY_PASSED: 600 epoch losses and 12 original weight states", flush=True)
    vr, vm, vy = data.source_data("val")
    val = {"clean": data.raw_cycles(vr, vm)}
    val.update({f"snr{k}": x for k,x in data.fixed_noisy_validation(vr,vm,noise_levels=(0,5,10,20,30),
                materialized_root=Path(deployment["strong_data_root"])/"generated/source").items()})
    selected_val = []
    for key, model in models.items():
        for condition, x in val.items():
            prediction = data.inference(model, x, torch.device("cuda"))
            m = independent_metrics(vy, prediction)
            expected = selections[key]["clean_f1"] if condition=="clean" else selections[key]["noisy_f1"][condition[3:]]
            close(m["macro_f1"], expected)
            selected_val.append({"model": key, "condition": condition, "metrics": m})
    print("SELECTED_SOURCE_VAL_REPRODUCED: 72 metric sets", flush=True)
    cached, comparisons, comparison_rows, gate_rows = {}, [], [], []
    native = read(OUT / "SUMMARY.json")
    for condition in conditions:
        level = None if condition=="clean" else int(condition[3:])
        root = Path(deployment["strong_data_root"])/"generated/target" if level in (0,5) else ROOT/"target_snr"
        raw, metadata = data.target_dev_from_mat(level, root)
        truth = np.array([r["label"] for r in metadata]); x = data.raw_cycles(raw, metadata)
        assert len(metadata)==2474 and len(raw)==88
        for method, seed in itertools.product(METHODS,seeds):
            key=f"{method}/seed{seed}"; saved=read(OUT/f"evaluations/{key}/target_dev_{condition}.json")
            assert saved["truth"]==truth.tolist() and saved["selected_source_epoch"]==selections[key]["epoch"]
            assert not saved["final_holdout_evaluated"] and not saved["target_used_for_training_or_selection"]
            if method=="nahsf":
                prediction, alpha, descriptor=predict_with_gate(models[key],x,torch.device("cuda"))
                gate=gate_summary(alpha,descriptor,truth)
                for k in gate:
                    close(gate[k],saved["gate"][k])
                gate_rows.append({"seed":seed,"condition":condition,**{k:gate[k] for k in ('mean','std','min','max','p10','p50','p90')}})
            else:
                prediction=data.inference(models[key],x,torch.device("cuda"))
            assert saved["predicted"]==prediction.tolist()
            m=independent_metrics(truth,prediction)
            for k in m:
                close(m[k],saved["metrics"][k])
            cached[method,seed,condition]=m
            # A′旧实验没有0/5 dB，只明确跳过这两档历史对照，不能拿其他条件填补。
            if method=='identity' and level in (0,5):
                continue
            baseline=(Path(deployment['original_pressure_test']) if method in ('lowpass3','lowpass4') and level in (0,5)
                      else Path(deployment['reference_roots'][method]))
            old=read(baseline/f"results/evaluations/{key}/target_dev_{condition}.json")
            assert old['truth']==truth.tolist()
            comparison_rows.append({'method':method,'seed':seed,'condition':condition,
                     'old_macro_f1_percent':100*old['metrics']['macro_f1'],'new_macro_f1_percent':100*m['macro_f1'],
                     'macro_f1_delta_pp':100*(m['macro_f1']-old['metrics']['macro_f1']),
                     'accuracy_delta_pp':100*(m['accuracy']-old['metrics']['accuracy'])})
        print(f"TARGET_REPLAYED {condition}: 12 predictions",flush=True)
    for method,condition in itertools.product(METHODS,conditions):
        scores=[100*cached[method,seed,condition]['macro_f1'] for seed in seeds]
        row=next(r for r in native['aggregates'] if r['method']==method and r['condition']==condition)
        close(row['macro_f1_percent']['mean'],np.mean(scores)); close(row['macro_f1_percent']['std'],np.std(scores,ddof=1))
        previous=[r for r in comparison_rows if r['method']==method and r['condition']==condition]
        if previous:
            comparisons.append({'method':method,'condition':condition,
                'macro_f1_delta_pp':stats([r['macro_f1_delta_pp'] for r in previous]),
                'accuracy_delta_pp':stats([r['accuracy_delta_pp'] for r in previous])})
    for row in native['paired_methods']:
        close(row['macro_f1_delta_pp']['values'],[100*(cached[row['left'],seed,row['condition']]['macro_f1']-
                     cached[row['right'],seed,row['condition']]['macro_f1']) for seed in seeds])
    csv_save(OUT/'BEFORE_AFTER_PER_SEED.csv',comparison_rows); csv_save(OUT/'NAHSF_GATE_PER_SEED.csv',gate_rows)
    write(OUT/'BEFORE_AFTER_SUMMARY.json',comparisons)
    for key,model in models.items():
        assert all(torch.equal(value.cpu(),checkpoints[key]['state'][name]) for name,value in model.state_dict().items())
        assert sha(OUT/f'models/{key}/best.pt')==weights[key]
    for p,expected in original.items():
        assert sha(p)==expected,p
    write(OUT/'VERIFICATION.json',{'status':'passed','code_commit':deployment['code_commit'],
        'trained_models':12,'epochs_per_model':50,'optimizer_updates_per_model':1200,
        'training_epoch_loss_exact_parity':600,'original_best_weight_states_exact_parity':12,
        'source_only_checkpoint_selections_verified':12,'selected_source_validation_metric_sets_reproduced':72,
        'selected_source_validation_results':selected_val,'target_prediction_vectors_exactly_reproduced':120,
        'target_metric_sets_independently_recomputed':120,'gate_summary_sets_reproduced':30,
        'paired_method_aggregate_groups_verified':60,'historical_paired_seed_rows':len(comparison_rows),
        'weights_sha256':weights,'selected_checkpoints':selections,'original_files_unchanged':len(original),
        'validation_saturation':saturation,'class_support':np.bincount(truth,minlength=6).tolist(),
        'final_holdout_evaluated':False,'verifier_sha256':sha(Path(__file__)),'verification_seconds':time.monotonic()-start})
    print('FOUR_MODELS_VERIFICATION_PASSED',flush=True)


if __name__=='__main__':
    main()
