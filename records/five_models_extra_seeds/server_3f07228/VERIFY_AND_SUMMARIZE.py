"""独立验证新750轮训练、90组源验证、150组新预测及六种子合并统计。"""
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
import five_models_extra_seeds as experiment
import four_models_expanded_val as common
import fixed_half_experiment as fixed
from v1_noise_consistency import fix_randomness
from v5_experiment import predict_with_gate, gate_summary
from evaluate_v4_strong_snr import frozen_ids, csv_write


def read(path):
    """直接读取原结果，不修改历史JSON或预测。"""
    return json.loads(Path(path).read_text())


def write(path, value):
    """独立核验输出只保存在本轮结果目录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    """逐字节保护代码、输入和旧结果。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def close(actual, expected):
    """只有浮点汇总末位允许误差，预测向量必须完全相同。"""
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def independent_metrics(truth, prediction):
    """使用sklearn独立复算；混淆矩阵行真实类别、列预测类别。"""
    cm = confusion_matrix(truth, prediction, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=range(6), zero_division=0)
    alarms = int(cm[0, 1:].sum())
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "normal_false_alarms": alarms,
            "normal_false_alarm_rate": alarms/int(cm[0].sum()), "missed_fault_as_normal": int(cm[1:, 0].sum())}


def main():
    """旧预测已通过历史GPU复现且SHA不变；新预测全部重新加载最佳权重验证。"""
    start = time.monotonic()
    assert read(ROOT / "RUN_STATUS.json")["status"] == "completed" and not (OUT / "VERIFICATION.json").exists()
    deploy, preflight, p = read(ROOT / "DEPLOYMENT_MANIFEST.json"), read(ROOT / "PREFLIGHT.json"), experiment.config()
    protected = read(ROOT / "ORIGINAL_FILES_SHA256.json")
    assert all(sha(path) == expected for path, expected in protected.items())
    assert all(sha(ROOT / "code" / path) == expected for path, expected in deploy["code_sha256"].items())
    four, half = Path(deploy["historical_four_root"]) / "results", Path(deploy["historical_fixed_root"]) / "results"
    assert all(read(folder / "VERIFICATION.json")["status"] == "passed" for folder in (four, half))
    models, selected, weights, selections = {}, {}, {}, []
    for method, seed in itertools.product(experiment.METHODS, experiment.ALL_SEEDS):
        root, protocol_name = experiment.result_root(method, seed, OUT, four, half)
        folder = root / f"models/{method}/seed{seed}"
        history, run = read(folder / "history.json"), read(folder / "run.json")
        assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
        assert all(run["protocol"][k] == p[k] for k in experiment.FAIR_KEYS)
        assert run["protocol"]["protocol"] == protocol_name and not run["target_used_for_training_or_selection"]
        for row in history:
            assert np.isfinite(row["train_loss_mean"]) and set(row["noisy_f1"]) == {"0", "5", "10", "20", "30"}
            close(row["noise_mean_f1"], np.mean(list(row["noisy_f1"].values())))
        eligible = [r for r in history if r["clean_f1"] >= max(v["clean_f1"] for v in history)-.01-1e-12]
        best = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
        assert best == run["selected"]
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        if seed in experiment.NEW_SEEDS:
            common.check_checkpoint(checkpoint, method, seed, protocol=p, version=experiment.VERSION)
            assert run["initial_cnn_sha256"] == preflight["initial_cnn_sha256"][str(seed)][method]
        elif method == "fixed_half":
            common.check_checkpoint(checkpoint, method, seed, protocol=fixed.config(), version=fixed.VERSION)
        else:
            common.check_checkpoint(checkpoint, method, seed)
        assert checkpoint["selected"] == best
        for name, value in checkpoint["state"].items():
            if name.endswith("num_batches_tracked"):
                assert int(value) == 24*best["epoch"]
        weights[f"{method}/seed{seed}"] = sha(folder / "best.pt")
        selected[method, seed] = best
        if seed in experiment.NEW_SEEDS:
            fix_randomness(seed)
            model = experiment.make_model(method).cuda()
            model.load_state_dict(checkpoint["state"], strict=True)
            model.eval()
            assert common.parameter_counts(model) == run["parameter_counts"]
            if method == "fixed_half":
                assert not any(k.startswith("shaping.") for k in checkpoint["state"])
            models[method, seed] = model
        selections.append({"method": method, "seed": seed, "group": "added" if seed in experiment.NEW_SEEDS else "original",
                           "selected_epoch": best["epoch"], "eligible_epochs": len(eligible),
                           "selected_clean_f1_percent": 100*best["clean_f1"],
                           "selected_five_snr_mean_percent": 100*best["noise_mean_f1"],
                           "five_snr_epochs_at_100": sum(r["noise_mean_f1"] >= 1-1e-12 for r in history)})
    csv_write(OUT / "VALIDATION_SELECTION.csv", selections)
    vr, vm, vy = data.source_data("val")
    val = {"clean": data.raw_cycles(vr, vm)}
    val.update({f"snr{k}": x for k, x in data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5, 10, 20, 30),
        materialized_root=Path(deploy["strong_data_root"]) / "generated/source").items()})
    source_replay = []
    for (method, seed), model in models.items():
        for condition, inputs in val.items():
            prediction = data.inference(model, inputs, torch.device("cuda"))
            m = independent_metrics(vy, prediction)
            best = selected[method, seed]
            close(m["macro_f1"], best["clean_f1"] if condition == "clean" else best["noisy_f1"][condition[3:]])
            source_replay.append({"method": method, "seed": seed, "condition": condition, "metrics": m})
    assert len(source_replay) == 90
    write(OUT / "SOURCE_VAL_REPLAY.json", source_replay)
    print("SOURCE_VAL_REPLAY_PASSED: 90 new sets", flush=True)
    from prepare import ROOT as DATA_ROOT, OUT as DATA_OUT
    metadata = read(DATA_OUT / "target_dev_clean_windows.json")
    frozen_ids(read(DATA_ROOT / "configs/splits_gate0_v1.json"), metadata)
    truth = np.array([r["label"] for r in metadata])
    native, cached, replay_count, gate_count = read(OUT / "SUMMARY.json"), {}, 0, 0
    conditions = ["clean" if k == "clean" else f"snr{k}" for k in p["target_conditions"]]
    for condition in conditions:
        level = None if condition == "clean" else int(condition[3:])
        target_root = Path(deploy["strong_data_root"]) / "generated/target" if level in (0, 5) else ROOT / "target_snr"
        raw, meta = data.target_dev_from_mat(level, target_root)
        assert meta == metadata
        inputs = data.raw_cycles(raw, metadata)
        for method, seed in itertools.product(experiment.METHODS, experiment.ALL_SEEDS):
            root, protocol_name = experiment.result_root(method, seed, OUT, four, half)
            result = read(root / f"evaluations/{method}/seed{seed}/target_dev_{condition}.json")
            experiment.validate_result(result, method, seed, condition, protocol_name)
            assert np.array_equal(truth, result["truth"])
            if seed in experiment.NEW_SEEDS:
                if method == "nahsf":
                    prediction, alpha, descriptor = predict_with_gate(models[method, seed], inputs, torch.device("cuda"))
                    actual_gate = gate_summary(alpha, descriptor, truth)
                    for key, value in actual_gate.items():
                        close(value, result["gate"][key])
                    gate_count += 1
                else:
                    prediction = data.inference(models[method, seed], inputs, torch.device("cuda"))
                assert np.array_equal(prediction, result["predicted"])
                replay_count += 1
            else:
                prediction = np.array(result["predicted"])
            m = independent_metrics(truth, prediction)
            for key, value in m.items():
                close(value, result["metrics"][key])
            cached[method, seed, condition] = (m, prediction)
        print(f"PREDICTION_REPLAY_PASSED {condition}: 15 new predictions + 15 protected historical metric sets", flush=True)
    assert len(native["aggregates"]) == 150 and len(native["paired_methods"]) == 300
    for row in native["aggregates"]:
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
            values = [100*cached[row["method"], s, row["condition"]][0][key] for s in row["seeds"]]
            close(row[key + "_percent"]["values"], values)
            close(row[key + "_percent"]["mean"], np.mean(values))
            close(row[key + "_percent"]["std"], np.std(values, ddof=1))
    for row in native["paired_methods"]:
        values = [100*(cached[row["left"], s, row["condition"]][0]["macro_f1"]-
                       cached[row["right"], s, row["condition"]][0]["macro_f1"]) for s in row["seeds"]]
        close(row["macro_f1_delta_pp"]["values"], values)
        close(row["macro_f1_delta_pp"]["mean"], np.mean(values))
        close(row["macro_f1_delta_pp"]["std"], np.std(values, ddof=1))
    # 核对公开CSV，不能只验证JSON而遗漏导出表的列含义。
    for filename, count in (("PER_SEED_RESULTS.csv", 300), ("PER_CLASS_PER_SEED.csv", 1800), ("PAIRED_PER_SEED.csv", 600)):
        with (OUT / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        for row in rows:
            seed, condition = int(row["seed"]), row["condition"]
            if filename == "PAIRED_PER_SEED.csv":
                (a, ap), (b, bp) = cached[row["left"], seed, condition], cached[row["right"], seed, condition]
                close(float(row["macro_f1_delta_pp"]), 100*(a["macro_f1"]-b["macro_f1"]))
                close(float(row["accuracy_delta_pp"]), 100*(a["accuracy"]-b["accuracy"]))
                assert int(row["left_only_correct"]) == int(((ap == truth) & (bp != truth)).sum())
                assert int(row["right_only_correct"]) == int(((bp == truth) & (ap != truth)).sum())
                assert int(row["prediction_disagreements"]) == int((ap != bp).sum())
            else:
                m = cached[row["method"], seed, condition][0]
                if filename == "PER_SEED_RESULTS.csv":
                    close(float(row["macro_f1_percent"]), 100*m["macro_f1"])
                    close(float(row["accuracy_percent"]), 100*m["accuracy"])
                    assert int(row["fault_as_healthy"]) == m["missed_fault_as_normal"]
                    assert int(row["healthy_as_fault"]) == m["normal_false_alarms"]
                else:
                    label = int(row["label"])
                    close(float(row["f1_percent"]), 100*m["per_class_f1"][label])
                    close(float(row["recall_percent"]), 100*m["per_class_recall"][label])
    assert all(sha(path) == expected for path, expected in native["input_results_sha256"].items())
    assert all(sha(path) == expected for path, expected in protected.items())
    # 数据池及元数据完全相同，沿用已核验的137个编号索引，保留原字节。
    (OUT / "DATA_FILE_INDEX.csv").write_bytes((half / "DATA_FILE_INDEX.csv").read_bytes())
    write(OUT / "VERIFICATION.json", {"status": "passed", "new_training_runs": 15, "new_training_epochs": 750,
        "historical_training_runs_reused": 15, "source_only_selections_independently_verified": 30,
        "new_source_validation_metric_sets_replayed": 90, "new_target_predictions_replayed": replay_count,
        "historical_target_metric_sets_independently_recomputed": 150, "new_gate_summaries_replayed": gate_count,
        "per_seed_rows_verified": 300, "per_class_rows_verified": 1800, "paired_per_seed_rows_verified": 600,
        "aggregate_rows_verified": 150, "paired_aggregate_rows_verified": 300,
        "weights_sha256": weights, "protected_original_files_checked": len(protected), "original_files_unchanged": True,
        "frozen_training_loop_models_and_configs_unchanged": True, "final_holdout_evaluated": False,
        "source_test_evaluated": False, "elapsed_seconds": time.monotonic()-start})
    print("FIVE_MODEL_EXTRA_SEEDS_VERIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
