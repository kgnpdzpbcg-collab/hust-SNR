"""独立复算V5训练选模、30组预测与门控，核对原LP3/LP4历史配对。"""
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
from v1_noise_consistency import fix_randomness
from v5_nahsf import NAHSFCNN, parameter_counts
from evaluate_v4_strong_snr import frozen_ids, CLASS_NAMES


def read(path):
    """使用服务器原始精度JSON，禁止从聊天表格反推指标。"""
    return json.loads(Path(path).read_text())


def sha(path):
    """校验权重、原数据、旧成绩及本轮代码没有被改写。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    """仅保存指标、标签和门控统计，不写出任何原始电流。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def close(left, right):
    """双精度指标只允许不超过1e-12的舍入差。"""
    np.testing.assert_allclose(left, right, atol=1e-12, rtol=0)


def stats(values):
    """样本标准差ddof=1，始终保留42/43/44的原数值。"""
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}


def csv_save(path, rows):
    """逐种子、逐类、门控和配对指标表。"""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_metrics(truth, pred):
    """sklearn复算六类F1/Recall，明确行是真实类、列是预测类。"""
    cm = confusion_matrix(truth, pred, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, pred, labels=range(6), zero_division=0)
    alarm = int(cm[0, 1:].sum())
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, pred)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "normal_false_alarms": alarm,
            "normal_false_alarm_rate": alarm / int(cm[0].sum()), "missed_fault_as_normal": int(cm[1:, 0].sum())}


def independent_gate(alpha, descriptor, truth):
    """从重放产生的门控值直接复算，与原gate_summary实现分开。"""
    assert alpha.shape == truth.shape and descriptor.shape == (len(truth), 3)
    assert np.isfinite(alpha).all() and np.isfinite(descriptor).all() and ((alpha >= 0) & (alpha <= 1)).all()
    p10, p50, p90 = np.quantile(alpha, [.1, .5, .9])
    return {"mean": float(alpha.mean()), "std": float(alpha.std()), "min": float(alpha.min()), "max": float(alpha.max()),
            "p10": float(p10), "p50": float(p50), "p90": float(p90),
            "by_true_class_mean": [float(alpha[truth == k].mean()) for k in range(6)],
            "descriptor_mean": descriptor.mean(axis=0).astype(float).tolist()}


def main():
    """重新载入三份最佳权重，评估过程含BatchNorm缓冲区在内不更新模型。"""
    started = time.monotonic()
    deployment, preflight = read(ROOT / "DEPLOYMENT_MANIFEST.json"), read(ROOT / "PREFLIGHT.json")
    assert read(ROOT / "RUN_STATUS.json")["status"] == "completed" and not (OUT / "VERIFICATION.json").exists()
    protected = read(ROOT / "ORIGINAL_FILES_SHA256.json")
    for path, expected in protected.items():
        assert sha(path) == expected, path
    for path, expected in deployment["code_sha256"].items():
        assert sha(ROOT / "code" / path) == expected, path
    seeds = [42, 43, 44]
    conditions = [None, 10, 15, 20, 25, 30, 35, 40, 5, 0]
    models, choices, weights, checkpoints = {}, {}, {}, {}
    for seed in seeds:
        folder = OUT / f"models/nahsf/seed{seed}"
        history, info = read(folder / "history.json"), read(folder / "run.json")
        assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
        assert info["epochs"] == 50 and info["batch"] == 32 and info["learning_rate"] == .001
        assert info["train_noise_levels_db"] == [10, 15, 20, 25, 30, 35, 40] and info["val_noise_levels_db"] == [10, 20, 30]
        assert info["initial_cnn_sha256"] == preflight["initial_cnn_sha256"][str(seed)]
        assert not info["target_used_for_training_or_checkpoint_selection"] and not info["final_holdout_evaluated"]
        for row in history:
            assert np.isfinite(row["train_loss_mean"]) and np.isfinite(row["gate_bias_grad_abs_mean"])
            assert set(row["noisy_f1"]) == {"10", "20", "30"}
            close(row["noise_mean_f1"], np.mean(list(row["noisy_f1"].values())))
        assert any(row["gate_bias_grad_abs_mean"] > 0 for row in history)
        maximum = max(r["clean_f1"] for r in history)
        eligible = [r for r in history if r["clean_f1"] >= maximum - .01 - 1e-12]
        chosen = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
        assert chosen == info["selected"]
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["selected"] == chosen and checkpoint["source_only"] and checkpoint["seed"] == seed
        assert checkpoint["code_family"] == "HUST-V5-NA-HSF-v0" and checkpoint["train_protocol"] == "V4_original_noise_10_to_40"
        for name, value in checkpoint["state"].items():
            if name.endswith("num_batches_tracked"):
                assert int(value) == chosen["epoch"] * 24
        fix_randomness(seed)
        model = NAHSFCNN().cuda()
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        assert parameter_counts(model) == info["parameter_counts"] == {"cnn": 25620, "spectral": 41, "total": 25661}
        models[seed], choices[str(seed)], weights[str(seed)], checkpoints[seed] = model, chosen, sha(folder / "best.pt"), checkpoint
    vr, vm, vy = data.source_data("val")
    val = {"clean": data.raw_cycles(vr, vm)}
    val.update({f"snr{k}": x for k, x in data.fixed_noisy_validation(vr, vm).items()})
    selected_val = []
    for seed, model in models.items():
        for name, x in val.items():
            prediction = data.inference(model, x, torch.device("cuda"))
            metric = independent_metrics(vy, prediction)
            expected = choices[str(seed)]["clean_f1"] if name == "clean" else choices[str(seed)]["noisy_f1"][name[3:]]
            close(metric["macro_f1"], expected)
            selected_val.append({"seed": seed, "condition": name, "metrics": metric})
    print("SOURCE_ONLY_SELECTION_AND_VAL_VERIFIED: 3 models, 12 sets", flush=True)
    scores, gates, predictions, gate_arrays = {}, {}, {}, {}
    per_seed, per_class, gate_rows = [], [], []
    for condition in conditions:
        name = "clean" if condition is None else f"snr{condition}"
        source = Path(deployment["strong_data_root"]) / "generated/target" if condition in (0, 5) else ROOT / "target_snr"
        raw, metadata = data.target_dev_from_mat(condition, source)
        ids = frozen_ids(read(ROOT / "code/configs/splits_gate0_v1.json"), metadata)
        truth = np.asarray([r["label"] for r in metadata])
        x = data.raw_cycles(raw, metadata)
        for seed, model in models.items():
            fix_randomness(seed)
            saved = read(OUT / f"evaluations/nahsf/seed{seed}/target_dev_{name}.json")
            assert saved["truth"] == truth.tolist() and saved["source_only_selection"] == choices[str(seed)]
            assert not saved["final_holdout_evaluated"] and not saved["target_used_for_training_or_checkpoint_selection"]
            pred, alpha, descriptor = [], [], []
            with torch.no_grad():
                for start in range(0, len(x), 128):
                    z, a, f = model(torch.from_numpy(x[start:start+128]).cuda(), return_descriptors=True)
                    pred.extend(z.argmax(1).cpu().tolist())
                    alpha.append(a.cpu().numpy())
                    descriptor.append(f.cpu().numpy())
            assert pred == saved["predicted"]
            metric = independent_metrics(truth, pred)
            for key in metric:
                close(metric[key], saved["metrics"][key])
            alpha, descriptor = np.concatenate(alpha), np.concatenate(descriptor)
            gate = independent_gate(alpha, descriptor, truth)
            for key in gate:
                close(gate[key], saved["gate"][key])
            scores[seed, name], gates[seed, name], predictions[seed, name], gate_arrays[seed, name] = metric, gate, np.array(pred), alpha
            per_seed.append({"seed": seed, "condition": name, "macro_f1_percent": 100*metric["macro_f1"],
                             "accuracy_percent": 100*metric["accuracy"], "fault_as_healthy": metric["missed_fault_as_normal"],
                             "healthy_as_fault": metric["normal_false_alarms"], "gate_alpha_mean": gate["mean"], "gate_alpha_std": gate["std"]})
            gate_rows.append({"seed": seed, "condition": name, **{k: gate[k] for k in ('mean', 'std', 'min', 'max', 'p10', 'p50', 'p90')},
                              **{f"descriptor{i+1}_mean": gate["descriptor_mean"][i] for i in range(3)}})
            for label, class_name in enumerate(CLASS_NAMES):
                per_class.append({"seed": seed, "condition": name, "label": label, "class_name": class_name,
                                  "support": int((truth == label).sum()), "f1_percent": 100*metric["per_class_f1"][label],
                                  "recall_percent": 100*metric["per_class_recall"][label], "gate_alpha_mean": gate["by_true_class_mean"][label],
                                  "fault_as_healthy": metric["confusion_matrix"][label][0] if label else ""})
        print(f"REPLAYED {name}: three predictions and gate summaries", flush=True)
    csv_save(OUT / "PER_SEED_RESULTS.csv", per_seed)
    csv_save(OUT / "PER_CLASS_PER_SEED.csv", per_class)
    csv_save(OUT / "GATE_PER_SEED.csv", gate_rows)
    labels = ["clean" if c is None else f"snr{c}" for c in conditions]
    aggregates, classes, comparisons, gate_changes = [], [], [], []
    pairs = list(csv.DictReader((OUT / "PAIRED_V4_COMPARISON.csv").open()))
    native = read(OUT / "SUMMARY.json")
    for name in labels:
        row = {"condition": name}
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
            row[key + "_percent"] = stats([100*scores[seed, name][key] for seed in seeds])
        for key in ("missed_fault_as_normal", "normal_false_alarms"):
            row[key] = stats([scores[seed, name][key] for seed in seeds])
        row["gate_alpha_mean"] = stats([gates[seed, name]["mean"] for seed in seeds])
        aggregates.append(row)
        for label, class_name in enumerate(CLASS_NAMES):
            classes.append({"condition": name, "label": label, "class_name": class_name,
                 "f1_percent": stats([100*scores[seed, name]["per_class_f1"][label] for seed in seeds]),
                 "recall_percent": stats([100*scores[seed, name]["per_class_recall"][label] for seed in seeds]),
                 "fault_as_healthy": stats([scores[seed, name]["confusion_matrix"][label][0] for seed in seeds]) if label else None})
        # 同一周期在Clean及加噪后的alpha差，按种子分别记录，不混合种子分布。
        gate_changes.append({"condition": name, "per_seed": [{"seed": seed,
                    "mean_alpha_change_from_clean": float((gate_arrays[seed, name]-gate_arrays[seed, 'clean']).mean()),
                    "mean_abs_alpha_change_from_clean": float(np.abs(gate_arrays[seed, name]-gate_arrays[seed, 'clean']).mean())} for seed in seeds]})
        for control in ("lowpass3", "lowpass4"):
            old_root = deployment["original_pressure_test"] if name in ("snr0", "snr5") else deployment["original_v4"]
            old = [read(Path(old_root) / f"results/evaluations/{control}/seed{seed}/target_dev_{name}.json") for seed in seeds]
            assert all(r['truth'] == truth.tolist() for r in old)
            delta = [100*(scores[seed, name]["macro_f1"]-r["metrics"]["macro_f1"]) for seed, r in zip(seeds, old)]
            for seed, historical, change in zip(seeds, old, delta):
                saved = next(r for r in pairs if r["seed"] == str(seed) and r["condition"] == name and r["vs"] == control)
                close(float(saved["delta_pp"]), change)
                p, q = predictions[seed, name], np.asarray(historical["predicted"])
                assert int(saved["v5_only_correct"]) == int(((p == truth) & (q != truth)).sum())
                assert int(saved["v4_only_correct"]) == int(((p != truth) & (q == truth)).sum())
            native_row = next(r for r in native["aggregates"] if r["condition"] == name and r["vs"] == control)
            close(native_row["mean_delta_pp"], np.mean(delta))
            close(native_row["sd_delta_pp"], np.std(delta, ddof=1))
            close(native_row["v5_mean_macro_f1"], row["macro_f1_percent"]["mean"])
            assert native_row["positive_seeds"] == sum(r > 0 for r in delta)
            comparisons.append({"condition": name, "vs": control, "macro_f1_delta_pp": stats(delta),
                  "accuracy_delta_pp": stats([100*(scores[seed, name]["accuracy"]-r["metrics"]["accuracy"]) for seed, r in zip(seeds, old)])})
    write(OUT / "EXTENDED_SUMMARY.json", {"seeds": seeds, "std_ddof": 1, "aggregates": aggregates,
           "per_class": classes, "paired_v4": comparisons, "gate_paired_change_from_clean": gate_changes})
    for seed, model in models.items():
        assert all(torch.equal(t.cpu(), checkpoints[seed]["state"][k]) for k, t in model.state_dict().items())
        assert sha(OUT / f"models/nahsf/seed{seed}/best.pt") == weights[str(seed)]
    for path, expected in protected.items():
        assert sha(path) == expected, path
    write(OUT / "VERIFICATION.json", {"status": "passed", "code_commit": deployment["code_commit"],
          "trained_models": 3, "epochs_per_model": 50, "optimizer_updates_per_model": 1200,
          "source_only_checkpoint_selections_verified": 3, "selected_source_val_metric_sets_reproduced": 12,
          "selected_source_validation_results": selected_val, "target_prediction_vectors_exactly_reproduced": 30,
          "target_metric_sets_independently_recomputed": 30, "gate_summary_sets_independently_recomputed": 30,
          "paired_baseline_seed_rows_verified": 60, "paired_baseline_aggregate_rows_verified": 20,
          "target_record_ids": ids, "target_cycles_per_condition": len(truth), "class_support": np.bincount(truth, minlength=6).tolist(),
          "weights_sha256": weights, "selected_checkpoints": choices, "original_files_unchanged": len(protected),
          "final_holdout_evaluated": False, "verifier_sha256": sha(Path(__file__)), "verification_seconds": time.monotonic()-started})
    print("V5_VERIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
