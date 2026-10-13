"""独立重载两模型权重，重算源域选模与实测预测，核对配对及受保护文件。"""
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
import fixed_half_experiment as experiment
import four_models_expanded_val as common
from v1_noise_consistency import fix_randomness
from evaluate_v4_strong_snr import frozen_ids


def read(path):
    """读取原始精度记录，独立验证不修改已有评估JSON。"""
    return json.loads(Path(path).read_text())


def write(path, value):
    """核验结果只保存在本轮目录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    """流式核对数据、旧结果和代码字节。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def close(actual, expected):
    """指标允许末位舍入差，不允许真实预测/配置差异。"""
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def independent_metrics(truth, prediction):
    """使用sklearn另算六类指标；矩阵行真实、列预测。"""
    cm = confusion_matrix(truth, prediction, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=range(6), zero_division=0)
    alarms = int(cm[0, 1:].sum())
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "normal_false_alarms": alarms,
            "normal_false_alarm_rate": alarms/int(cm[0].sum()), "missed_fault_as_normal": int(cm[1:, 0].sum())}


def csv_write(path, rows):
    """导出原始记录编号与处理范围，避免只凭周期数描述使用的数据。"""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """原权重和输入必须不变，选模仅来自源域，60组实测预测逐周期重现。"""
    start = time.monotonic()
    assert read(ROOT / "RUN_STATUS.json")["status"] == "completed" and not (OUT / "VERIFICATION.json").exists()
    deploy, preflight, p = read(ROOT / "DEPLOYMENT_MANIFEST.json"), read(ROOT / "PREFLIGHT.json"), experiment.config()
    protected = read(ROOT / "ORIGINAL_FILES_SHA256.json")
    assert all(sha(path) == expected for path, expected in protected.items())
    assert all(sha(ROOT / "code" / path) == expected for path, expected in deploy["code_sha256"].items())
    reference = Path(deploy["reference_root"]) / "results"
    models, chosen, weights, saturation = {}, {}, {}, []
    for method, seed in itertools.product(("fixed_half", "nahsf"), p["seeds"]):
        folder = (OUT if method == "fixed_half" else reference) / f"models/{method}/seed{seed}"
        history, run = read(folder / "history.json"), read(folder / "run.json")
        assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
        assert all(run["protocol"][k] == p[k] for k in experiment.FAIR_KEYS)
        assert run["initial_cnn_sha256"] == preflight["initial_cnn_sha256"][str(seed)]
        for row in history:
            assert np.isfinite(row["train_loss_mean"]) and set(row["noisy_f1"]) == {"0", "5", "10", "20", "30"}
            close(row["noise_mean_f1"], np.mean(list(row["noisy_f1"].values())))
        eligible = [r for r in history if r["clean_f1"] >= max(v["clean_f1"] for v in history)-.01-1e-12]
        best = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
        assert best == run["selected"]
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        if method == "fixed_half":
            common.check_checkpoint(checkpoint, method, seed, protocol=p, version=experiment.VERSION)
        else:
            common.check_checkpoint(checkpoint, method, seed)
        assert checkpoint["selected"] == best
        for name, value in checkpoint["state"].items():
            if name.endswith("num_batches_tracked"):
                assert int(value) == 24*best["epoch"]
        fix_randomness(seed)
        model = (experiment.make_model(method) if method == "fixed_half" else common.make_model(method)).cuda()
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        if method == "fixed_half":
            assert not any(name.startswith("shaping.") for name in checkpoint["state"])
            assert common.parameter_counts(model) == {"cnn": 25620, "spectral": 0, "total": 25620}
        models[method, seed], chosen[method, seed], weights[f"{method}/seed{seed}"] = model, best, sha(folder / "best.pt")
        saturation.append({"method": method, "seed": seed, "selected_epoch": best["epoch"], "eligible_epochs": len(eligible),
              "three_snr_epochs_at_100": sum(r["original_three_snr_mean_f1"] >= 1-1e-12 for r in history),
              "five_snr_epochs_at_100": sum(r["noise_mean_f1"] >= 1-1e-12 for r in history),
              "five_snr_max_percent": 100*max(r["noise_mean_f1"] for r in history)})
    csv_write(OUT / "VALIDATION_SELECTION.csv", saturation)
    vr, vm, vy = data.source_data("val")
    val = {"clean": data.raw_cycles(vr, vm)}
    val.update({f"snr{k}": x for k, x in data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5, 10, 20, 30),
          materialized_root=Path(deploy["strong_data_root"]) / "generated/source").items()})
    selected_source = []
    for (method, seed), model in models.items():
        for condition, inputs in val.items():
            prediction = data.inference(model, inputs, torch.device("cuda"))
            m = independent_metrics(vy, prediction)
            expected = chosen[method, seed]["clean_f1"] if condition == "clean" else chosen[method, seed]["noisy_f1"][condition[3:]]
            close(m["macro_f1"], expected)
            selected_source.append({"method": method, "seed": seed, "condition": condition, "metrics": m})
    write(OUT / "SOURCE_VAL_REPLAY.json", selected_source)
    print("SOURCE_VAL_REPLAY_PASSED: 36 sets", flush=True)
    from prepare import ROOT as DATA_ROOT, OUT as DATA_OUT
    metadata = read(DATA_OUT / "target_dev_clean_windows.json")
    split = read(DATA_ROOT / "configs/splits_gate0_v1.json")
    frozen_ids(split, metadata)
    truth = np.array([row["label"] for row in metadata])
    assert len(truth) == 2474 and len({row["record_id"] for row in metadata}) == 88
    native, cached = read(OUT / "SUMMARY.json"), {}
    conditions = ["clean" if k == "clean" else f"snr{k}" for k in p["target_conditions"]]
    for condition in conditions:
        level = None if condition == "clean" else int(condition[3:])
        root = Path(deploy["strong_data_root"]) / "generated/target" if level in (0, 5) else ROOT / "target_snr"
        raw, actual_meta = data.target_dev_from_mat(level, root)
        assert actual_meta == metadata
        inputs = data.raw_cycles(raw, metadata)
        for (method, seed), model in models.items():
            result = read((OUT if method == "fixed_half" else reference) / f"evaluations/{method}/seed{seed}/target_dev_{condition}.json")
            prediction = data.inference(model, inputs, torch.device("cuda"))
            assert np.array_equal(result["truth"], truth) and np.array_equal(result["predicted"], prediction)
            m = independent_metrics(truth, prediction)
            for key, value in m.items():
                close(value, result["metrics"][key])
            cached[method, seed, condition] = (m, prediction)
        print(f"PREDICTION_REPLAY_PASSED {condition}: 6 sets", flush=True)
    for row in native["aggregates"]:
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
            values = [100*cached[row["method"], seed, row["condition"]][0][key] for seed in p["seeds"]]
            close(row[key + "_percent"]["values"], values)
            close(row[key + "_percent"]["mean"], np.mean(values))
            close(row[key + "_percent"]["std"], np.std(values, ddof=1))
    with (OUT / "PAIRED_V5_COMPARISON.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 30
    for row in pairs:
        seed, condition = int(row["seed"]), row["condition"]
        (a, ap), (b, bp) = cached["nahsf", seed, condition], cached["fixed_half", seed, condition]
        close(float(row["v5_minus_fixed_macro_f1_pp"]), 100*(a["macro_f1"]-b["macro_f1"]))
        close(float(row["v5_minus_fixed_accuracy_pp"]), 100*(a["accuracy"]-b["accuracy"]))
        assert int(row["v5_only_correct"]) == int(((ap == truth) & (bp != truth)).sum())
        assert int(row["fixed_only_correct"]) == int(((bp == truth) & (ap != truth)).sum())
        assert int(row["prediction_disagreements"]) == int((ap != bp).sum())
    for row in native["paired_v5_minus_fixed"]:
        values = [100*(cached["nahsf", s, row["condition"]][0]["macro_f1"]-
                      cached["fixed_half", s, row["condition"]][0]["macro_f1"]) for s in p["seeds"]]
        close(row["v5_minus_fixed_macro_f1_pp"]["values"], values)
        close(row["v5_minus_fixed_macro_f1_pp"]["mean"], np.mean(values))
        close(row["v5_minus_fixed_macro_f1_pp"]["std"], np.std(values, ddof=1))
    for path, expected in native["input_results_sha256"].items():
        assert sha(path) == expected
    assert all(sha(path) == expected for path, expected in protected.items())
    # 编号与周期数导出，不包含原始电流。所有窗口沿用Gate0冻结元数据。
    indices = []
    for part in ("train", "val"):
        _, meta, _ = data.source_data(part)
        for rid in sorted({row["record_id"] for row in meta}):
            rows = [r for r in meta if r["record_id"] == rid]
            indices.append({"domain": "simulation", "partition": part, "record_id": rid,
                            "cycles": len(rows), "label": rows[0]["label"]})
    for rid in sorted({row["record_id"] for row in metadata}):
        rows = [r for r in metadata if r["record_id"] == rid]
        indices.append({"domain": "experimental", "partition": "dev", "record_id": rid,
                        "cycles": len(rows), "label": rows[0]["label"]})
    assert len(indices) == 49+88
    csv_write(OUT / "DATA_FILE_INDEX.csv", indices)
    write(OUT / "VERIFICATION.json", {"status": "passed", "completed_training_runs": 3, "training_epochs": 150,
          "v5_training_runs_reused": 3, "replayed_target_predictions": 60, "replayed_source_validation_sets": 36,
          "paired_per_seed_rows_verified": 30, "paired_aggregate_rows_verified": 10,
          "source_only_selection_independently_verified": True, "fixed_alpha_no_spectral_parameters": True,
          "best_weights_sha256": weights, "protected_original_files_checked": len(protected),
          "original_files_unchanged": True, "code_files_unchanged": True, "final_holdout_evaluated": False,
          "source_test_evaluated": False, "elapsed_seconds": time.monotonic()-start})
    print("FIXED_HALF_VERIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
