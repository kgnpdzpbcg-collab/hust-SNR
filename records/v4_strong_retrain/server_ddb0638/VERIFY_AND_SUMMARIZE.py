"""独立核查强噪声重训、选模、120组实测预测，并生成完整配对指标。"""
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
from v4_hpasf import HPASFCNN, METHODS, counts_by_module
from v4_strong_noise_experiment import config, check_checkpoint
from evaluate_v4_strong_snr import CLASS_NAMES, frozen_ids


def read(path):
    """始终读取服务器实际结果，禁止根据聊天中四舍五入数值计算。"""
    return json.loads(Path(path).read_text())


def write(path, value):
    """JSON保留双精度原数值，表格只负责显示。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    """字节指纹用于确认原文件和本轮权重未在核查过程中被改写。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right):
    """独立指标计算仅允许1e-12的浮点舍入差。"""
    np.testing.assert_allclose(left, right, atol=1e-12, rtol=0)


def stats(values):
    """三个种子全部保留，差值先逐种子配对，再计算样本标准差。"""
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}


def save_csv(path, rows):
    """逐种子、逐类别或整记录SNR数量表。"""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_metrics(truth, prediction):
    """使用sklearn复算六类指标，并明确健康误报/故障漏报的矩阵方向。"""
    cm = confusion_matrix(truth, prediction, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=range(6), zero_division=0)
    alarm = int(cm[0, 1:].sum())
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "normal_false_alarms": alarm,
            "normal_false_alarm_rate": alarm / int(cm[0].sum()), "missed_fault_as_normal": int(cm[1:, 0].sum())}


def main():
    """先查真实训练日志与源验证选模，再重新载入模型重放全部实测预测。"""
    started = time.monotonic()
    assert read(ROOT / "RUN_STATUS.json")["status"] == "completed"
    assert not (OUT / "VERIFICATION.json").exists()
    deployment, preflight, protocol = read(ROOT / "DEPLOYMENT_MANIFEST.json"), read(ROOT / "PREFLIGHT.json"), config()
    originals = read(ROOT / "ORIGINAL_FILES_SHA256.json")
    for path, expected in originals.items():
        assert sha(path) == expected, path
    for path, expected in deployment["code_sha256"].items():
        assert sha(ROOT / "code" / path) == expected, path
    old, pressure, strong = map(Path, [deployment["original_v4"], deployment["original_pressure_test"], deployment["strong_data_root"]])
    seeds = protocol["seeds"]
    conditions = [None if c == "clean" else c for c in protocol["target_conditions"]]
    models, weights, choices, checkpoints, schedules, gates = {}, {}, {}, {}, [], {}
    train_raw, train_meta, _ = data.source_data("train")
    for seed in seeds:
        for epoch in range(1, 51):
            # 独立展开原随机流，检查每轮的九档分配与记录数量。
            rng = np.random.default_rng(seed + 15000 + epoch)
            levels = [int(rng.choice(protocol["training_noise_levels_db"])) for _ in sorted(train_raw)]
            schedules += [{"seed": seed, "epoch": epoch, "snr_db": level, "raw_record_count": levels.count(level)}
                          for level in protocol["training_noise_levels_db"]]
    for method, seed in itertools.product(METHODS, seeds):
        key = f"{method}/seed{seed}"
        folder = OUT / "models" / key
        history, info = read(folder / "history.json"), read(folder / "run.json")
        assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
        assert info["epochs"] == 50 and info["batch_size"] == 32 and info["lr"] == .001
        assert info["initial_cnn_sha256"] == preflight["initial_cnn_sha256"][str(seed)][method]
        assert info["train_protocol"] == "V4_strong_noise_v1" and not info["target_used_for_training_or_selection"]
        for row in history:
            assert np.isfinite(row["train_loss_mean"])
            assert set(row["noisy_f1"]) == {"0", "5", "10", "20", "30"}
            close(row["noise_mean_f1"], np.mean(list(row["noisy_f1"].values())))
            expected = {str(r["snr_db"]): r["raw_record_count"] for r in schedules if r["seed"] == seed and r["epoch"] == row["epoch"]}
            assert row["train_snr_record_counts"] == expected and sum(expected.values()) == 34
        maximum_clean = max(r["clean_f1"] for r in history)
        eligible = [r for r in history if r["clean_f1"] >= maximum_clean - .01 - 1e-12]
        chosen = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
        assert chosen == info["selected"]
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        check_checkpoint(checkpoint, protocol)
        assert checkpoint["selected"] == chosen and checkpoint["mode"] == method and checkpoint["seed"] == seed and checkpoint["source_only"]
        for name, value in checkpoint["state"].items():
            if name.endswith("num_batches_tracked"):
                assert int(value) == chosen["epoch"]*24
        fix_randomness(seed)
        model = HPASFCNN(method).cuda()
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        assert counts_by_module(model) == info["parameter_counts"]
        models[key], choices[key], checkpoints[key], weights[key] = model, chosen, checkpoint, sha(folder / "best.pt")
        if method in ("static4", "hpasf"):
            gates[key] = {"gate_logit": float(checkpoint["state"]["shaping.gate_logit"]),
                          "base_gain4": float(torch.sigmoid(checkpoint["state"]["shaping.gate_logit"]))}
    save_csv(OUT / "TRAINING_NOISE_COUNTS.csv", schedules)
    print("50_EPOCH_TRAINING_AND_SOURCE_ONLY_SELECTION_VERIFIED", flush=True)
    vr, vm, vy = data.source_data("val")
    validation = {"clean": data.raw_cycles(vr, vm)}
    validation.update({f"snr{level}": value for level, value in data.fixed_noisy_validation(
        vr, vm, noise_levels=tuple(protocol["validation_noise_levels_db"]), materialized_root=strong / "generated/source").items()})
    selected_val = []
    for key, model in models.items():
        for condition, values in validation.items():
            pred = data.inference(model, values, torch.device("cuda"))
            metric = independent_metrics(vy, pred)
            expected = choices[key]["clean_f1"] if condition == "clean" else choices[key]["noisy_f1"][condition[3:]]
            close(metric["macro_f1"], expected)
            selected_val.append({"model": key, "condition": condition, "metrics": metric})
    print("SELECTED_SOURCE_VAL_REPRODUCED: 72 metric sets", flush=True)
    cached, aggregates, classes, pairs, comparisons, per_seed, per_class, before_after = {}, [], [], [], [], [], [], []
    for level in conditions:
        condition = "clean" if level is None else f"snr{level}"
        raw, metadata = data.target_dev_from_mat(level, ROOT / "target_snr")
        ids = frozen_ids(read(ROOT / "code/configs/splits_gate0_v1.json"), metadata)
        truth = np.array([r["label"] for r in metadata])
        inputs = data.raw_cycles(raw, metadata)
        for method, seed in itertools.product(METHODS, seeds):
            key = f"{method}/seed{seed}"
            saved = read(OUT / f"evaluations/{key}/target_dev_{condition}.json")
            assert saved["truth"] == truth.tolist() and saved["source_only_selection"] == choices[key]
            assert saved["data_source"] == "original_MAT" and not saved["final_holdout_evaluated"]
            pred = data.inference(models[key], inputs, torch.device("cuda"))
            assert saved["predicted"] == pred.tolist()
            metrics = independent_metrics(truth, pred)
            for name in metrics:
                close(metrics[name], saved["metrics"][name])
            cached[method, seed, condition] = metrics
            base = {"method": method, "seed": seed, "condition": condition}
            per_seed.append({**base, "macro_f1_percent": 100*metrics["macro_f1"], "accuracy_percent": 100*metrics["accuracy"],
                             "fault_as_healthy": metrics["missed_fault_as_normal"], "healthy_as_fault": metrics["normal_false_alarms"]})
            for label, class_name in enumerate(CLASS_NAMES):
                per_class.append({**base, "label": label, "class_name": class_name, "support": int((truth == label).sum()),
                                  "f1_percent": 100*metrics["per_class_f1"][label], "recall_percent": 100*metrics["per_class_recall"][label],
                                  "fault_as_healthy": metrics["confusion_matrix"][label][0] if label else ""})
            baseline_root = pressure if level in (0, 5) else old
            baseline = read(baseline_root / f"results/evaluations/{key}/target_dev_{condition}.json")
            assert baseline["truth"] == truth.tolist()
            b = baseline["metrics"]
            before_after.append({**base, "old_macro_f1_percent": 100*b["macro_f1"], "new_macro_f1_percent": 100*metrics["macro_f1"],
                                 "macro_f1_delta_pp": 100*(metrics["macro_f1"]-b["macro_f1"]),
                                 "old_accuracy_percent": 100*b["accuracy"], "new_accuracy_percent": 100*metrics["accuracy"],
                                 "accuracy_delta_pp": 100*(metrics["accuracy"]-b["accuracy"])})
        print(f"TARGET {condition}: 12 prediction vectors reproduced", flush=True)
    native = read(OUT / "SUMMARY.json")
    labels = ["clean" if c is None else f"snr{c}" for c in conditions]
    for method, condition in itertools.product(METHODS, labels):
        scores = [cached[method, seed, condition] for seed in seeds]
        row = {"method": method, "condition": condition}
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
            row[metric + "_percent"] = stats([100*r[metric] for r in scores])
        for metric in ("missed_fault_as_normal", "normal_false_alarms"):
            row[metric] = stats([r[metric] for r in scores])
        original_summary = next(r for r in native["aggregates"] if r["method"] == method and r["condition"] == condition)
        close(row["macro_f1_percent"]["mean"], original_summary["macro_f1_mean_percent"])
        close(row["macro_f1_percent"]["std"], original_summary["macro_f1_sd_percent"])
        aggregates.append(row)
        for label, name in enumerate(CLASS_NAMES):
            classes.append({"method": method, "condition": condition, "label": label, "class_name": name,
                            "support": int((truth == label).sum()),
                            "f1_percent": stats([100*r["per_class_f1"][label] for r in scores]),
                            "recall_percent": stats([100*r["per_class_recall"][label] for r in scores]),
                            "fault_as_healthy": stats([r["confusion_matrix"][label][0] for r in scores]) if label else None})
        previous = [r for r in before_after if r["method"] == method and r["condition"] == condition]
        comparisons.append({"method": method, "condition": condition,
                            "macro_f1_delta_pp": stats([r["macro_f1_delta_pp"] for r in previous]),
                            "accuracy_delta_pp": stats([r["accuracy_delta_pp"] for r in previous])})
    for right, left in itertools.combinations(METHODS, 2):
        for condition in labels:
            delta = {"left": left, "right": right, "condition": condition}
            for metric in ("macro_f1", "accuracy"):
                delta[metric + "_delta_pp"] = stats([100*(cached[left, seed, condition][metric]-cached[right, seed, condition][metric]) for seed in seeds])
            pairs.append(delta)
    for row in native["paired_vs_lowpass3"]:
        actual = next(r for r in pairs if r["left"] == row["method_minus_lowpass3"] and r["right"] == "lowpass3" and r["condition"] == row["condition"])
        close(actual["macro_f1_delta_pp"]["values"], row["paired_deltas_pp"])
    write(OUT / "EXTENDED_SUMMARY.json", {"status": "passed", "std_ddof": 1, "seeds": seeds,
          "aggregates": aggregates, "per_class": classes, "paired_methods": pairs, "new_minus_original": comparisons})
    save_csv(OUT / "FULL_PER_SEED_RESULTS.csv", per_seed)
    save_csv(OUT / "PER_CLASS_PER_SEED.csv", per_class)
    save_csv(OUT / "BEFORE_AFTER_PER_SEED.csv", before_after)
    for key, model in models.items():
        assert all(torch.equal(value.cpu(), checkpoints[key]["state"][name]) for name, value in model.state_dict().items())
        assert sha(OUT / f"models/{key}/best.pt") == weights[key]
    for path, expected in originals.items():
        assert sha(path) == expected, path
    write(OUT / "VERIFICATION.json", {"status": "passed", "code_commit": deployment["code_commit"],
          "trained_models": 12, "epochs_per_model": 50, "optimizer_updates_per_model": 1200,
          "training_snr_counts_verified_epochs": 600, "source_only_checkpoint_selections_verified": 12,
          "selected_source_validation_metric_sets_reproduced": 72, "selected_source_validation_results": selected_val,
          "target_prediction_vectors_exactly_reproduced": 120, "target_metric_sets_independently_recomputed": 120,
          "target_record_ids": ids, "target_cycles_per_condition": len(truth),
          "class_support": np.bincount(truth, minlength=6).tolist(), "aggregate_groups_verified": 40,
          "before_after_paired_groups_verified": 40, "weights_sha256": weights, "selected_checkpoints": choices,
          "learned_scalar_gates": gates, "original_files_unchanged": len(originals), "final_holdout_evaluated": False,
          "verifier_sha256": sha(Path(__file__)), "verification_seconds": time.monotonic()-started})
    print("VERIFICATION_PASSED: 12 training runs, 72 source-val sets, 120 target sets", flush=True)


if __name__ == "__main__":
    main()
