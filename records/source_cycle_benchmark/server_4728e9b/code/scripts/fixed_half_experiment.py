"""固定半增益消融：独立输出、复用四模型训练/评估流程，与已有V5配对。

只新训练固定α模型的三个种子；V5使用五档源验证选模实验的冻结结果。
旧三档选模快照仅保留作审计，不进入本轮实测比较。
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
from pathlib import Path
import numpy as np
import four_models_expanded_val as common
from fixed_half_fusion import FixedHalfCNN
from evaluate_v4_strong_snr import read, write, csv_write, stats, CLASS_NAMES

CONFIG = Path(__file__).resolve().parents[1] / "configs/fixed_half_vs_v5_v1.json"
METHOD = "fixed_half"
VERSION = "HUST-fixed-half-fusion-v1"
FAIR_KEYS = ("seeds", "epochs", "batch_size", "learning_rate", "training_noise_levels_db",
             "validation_noise_levels_db", "target_conditions", "loss", "selection")


def config():
    """校验与V5相同的训练和选模协议，禁止改变噪声、预算或超参数。"""
    value, reference = read(CONFIG), common.config()
    if any(value[k] != reference[k] for k in FAIR_KEYS):
        raise ValueError("固定半增益协议与V5冻结协议不一致")
    if value["methods"] != [METHOD] or value["fixed_alpha"] != .5 or value["use_descriptors"] or value["use_gate_mlp"]:
        raise ValueError("固定半增益必须无描述符、无门控且α=0.5")
    if value["protocol"] != "fixed_half_train10_40_val0_5_10_20_30_v1":
        raise ValueError("固定半增益需要独立协议编号")
    return value


def make_model(method):
    """仅支持本轮固定模型，不扩大原四模型入口的可选模型列表。"""
    if method != METHOD:
        raise ValueError(method)
    return FixedHalfCNN()


def summarize(output, v5_results):
    """同种子、同周期逐一对照V5；保存均值、样本标准差和逐种子差值。

    V5结果必须来自五档源验证协议。预测只能用于事后比较，不能用于
    训练或最佳轮次选择；三种子差值不自动解释为统计显著性。
    """
    p = config()
    conditions = ["clean" if k == "clean" else f"snr{k}" for k in p["target_conditions"]]
    cached, rows, classes, pairs, hashes = {}, [], [], [], {}
    truth = None
    for method, seed, condition in itertools.product((METHOD, "nahsf"), p["seeds"], conditions):
        root = output if method == METHOD else v5_results
        run_path = root / f"models/{method}/seed{seed}/run.json"
        run = read(run_path)
        if any(run["protocol"][k] != p[k] for k in FAIR_KEYS):
            raise ValueError("V5与固定模型的训练/选模配置不一致")
        path = root / f"evaluations/{method}/seed{seed}/target_dev_{condition}.json"
        result = read(path)
        expected_protocol = p["protocol"] if method == METHOD else common.config()["protocol"]
        if (result["protocol"] != expected_protocol or result["method"] != method or result["seed"] != seed or
                result["condition"] != condition or result["windows"] != 2474 or result["record_families"] != 88 or
                result["target_used_for_training_or_selection"] is not False or result["final_holdout_evaluated"] is not False or
                result["selected_source_epoch"] != run["selected"]["epoch"]):
            raise ValueError("比较结果来源不一致")
        if truth is None:
            truth = result["truth"]
        if result["truth"] != truth:
            raise ValueError("比较结果的周期顺序不一致")
        cached[method, seed, condition] = result
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(run_path)] = hashlib.sha256(run_path.read_bytes()).hexdigest()
        base = {"method": method, "seed": seed, "condition": condition}
        metric = result["metrics"]
        rows.append({**base, "selected_source_epoch": result["selected_source_epoch"],
                     "macro_f1_percent": 100*metric["macro_f1"], "accuracy_percent": 100*metric["accuracy"],
                     "fault_as_healthy": metric["missed_fault_as_normal"], "healthy_as_fault": metric["normal_false_alarms"]})
        for label, name in enumerate(CLASS_NAMES):
            classes.append({**base, "label": label, "class_name": name, "f1_percent": 100*metric["per_class_f1"][label],
                            "recall_percent": 100*metric["per_class_recall"][label],
                            "fault_as_healthy": metric["confusion_matrix"][label][0] if label else ""})
    aggregates, paired_stats = [], []
    for method, condition in itertools.product((METHOD, "nahsf"), conditions):
        row = {"method": method, "condition": condition}
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
            row[key + "_percent"] = stats([100*cached[method, seed, condition]["metrics"][key] for seed in p["seeds"]])
        aggregates.append(row)
    for condition in conditions:
        deltas = []
        for seed in p["seeds"]:
            left, right = cached["nahsf", seed, condition], cached[METHOD, seed, condition]
            delta = 100*(left["metrics"]["macro_f1"] - right["metrics"]["macro_f1"])
            deltas.append(delta)
            a, b, y = np.asarray(left["predicted"]), np.asarray(right["predicted"]), np.asarray(truth)
            pairs.append({"seed": seed, "condition": condition, "v5_minus_fixed_macro_f1_pp": delta,
                          "v5_minus_fixed_accuracy_pp": 100*(left["metrics"]["accuracy"]-right["metrics"]["accuracy"]),
                          "v5_only_correct": int(((a == y) & (b != y)).sum()),
                          "fixed_only_correct": int(((b == y) & (a != y)).sum()),
                          "prediction_disagreements": int((a != b).sum())})
        paired_stats.append({"condition": condition, "v5_minus_fixed_macro_f1_pp": stats(deltas)})
    write(output / "SUMMARY.json", {"protocol": p, "aggregates": aggregates, "paired_v5_minus_fixed": paired_stats,
          "std_ddof": 1, "v5_reference_results": str(v5_results), "input_results_sha256": hashes})
    csv_write(output / "PER_SEED_RESULTS.csv", rows)
    csv_write(output / "PER_CLASS_PER_SEED.csv", classes)
    csv_write(output / "PAIRED_V5_COMPARISON.csv", pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 43, 44), default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strong-data-root", type=Path)
    parser.add_argument("--snr-root", type=Path)
    parser.add_argument("--v5-results", type=Path)
    args = parser.parse_args()
    if args.stage in ("train", "evaluate") and args.strong_data_root is None:
        parser.error("需要固定0/5 dB验证及实测数据目录")
    if args.stage == "evaluate" and args.snr_root is None:
        parser.error("需要原10–40 dB实测目录")
    if args.stage == "summarize":
        if args.v5_results is None:
            parser.error("需要五档源验证选模的V5结果目录")
        summarize(args.output, args.v5_results)
    else:
        options = dict(protocol=config(), model_factory=make_model, version=VERSION)
        for seed in args.seeds:
            if args.stage == "train":
                common.train(METHOD, seed, args.output, args.strong_data_root, **options)
            else:
                common.evaluate(METHOD, seed, args.output, args.snr_root, args.strong_data_root, **options)
