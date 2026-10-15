"""五模型追加种子2024/2025/2026：原训练循环不变，单独保存并合并六种子。

本轮仅新增15次训练。42/43/44来自已核验的四模型扩大验证和固定α实验，
保留原成绩和权重。所有模型共用同一数据池、50轮训练及五档源验证选模。
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
from pathlib import Path
import numpy as np
import four_models_expanded_val as common
import fixed_half_experiment as fixed
from evaluate_v4_strong_snr import read, write, csv_write, CLASS_NAMES

CONFIG = Path(__file__).resolve().parents[1] / "configs/five_models_extra_seeds_v1.json"
METHODS = (*common.METHODS, fixed.METHOD)
NEW_SEEDS = (2024, 2025, 2026)
OLD_SEEDS = (42, 43, 44)
ALL_SEEDS = (*OLD_SEEDS, *NEW_SEEDS)
VERSION = "HUST-five-models-extra-seeds-v1"
FAIR_KEYS = tuple(k for k in fixed.FAIR_KEYS if k != "seeds")


def config():
    """仅扩展种子和方法清单，其余冻结配置必须与最近两次实验一致。"""
    p, original = read(CONFIG), common.config()
    if (p["methods"] != list(METHODS) or p["seeds"] != list(NEW_SEEDS) or
            p["original_seeds"] != list(OLD_SEEDS) or p["combined_seeds"] != list(ALL_SEEDS) or
            p["protocol"] != "five_models_extra_seeds2024_2025_2026_v1" or
            p["fixed_alpha"] != .5 or p["target_used_for_training_or_selection"] or p["final_holdout_evaluated"] or
            any(p[k] != original[k] for k in FAIR_KEYS)):
        raise ValueError("追加种子配置不得更改冻结训练、验证、数据或选模规则")
    # 原固定α入口也需通过其自身冻结校验，防止静默换成其他融合方法。
    fixed.config()
    return p


def make_model(method):
    """直接构造原五种模型；不一次构造全部模型，保持CNN初始化随机流。"""
    return fixed.make_model(method) if method == fixed.METHOD else common.make_model(method)


def statistics(values):
    """三个新增/原种子，或六个合并种子：保留全部值，标准差ddof=1。

    原压力评估stats函数只接受三个种子；本入口独立支持六个，不修改旧函数。
    """
    if len(values) not in (3, 6) or not np.isfinite(values).all():
        raise ValueError("统计必须包含完整的三个或六个有限种子结果")
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": list(values)}


def result_root(method, seed, output, four_results, fixed_results):
    """按模型/种子选择实际结果目录，禁止用旧三档验证或强噪声训练结果。"""
    if seed in NEW_SEEDS:
        return output, config()["protocol"]
    if seed not in OLD_SEEDS:
        raise ValueError(f"未声明的种子：{seed}")
    return (fixed_results, fixed.config()["protocol"]) if method == fixed.METHOD else (four_results, common.config()["protocol"])


def validate_result(result, method, seed, condition, protocol):
    """核验来源、周期规模及源域选模标记，不把其他实验的分数混入汇总。"""
    expected = {"method": method, "seed": seed, "condition": condition, "protocol": protocol,
                "record_families": 88, "windows": 2474, "target_used_for_training_or_selection": False,
                "final_holdout_evaluated": False}
    if (any(result.get(k) != v for k, v in expected.items()) or
            len(result.get("truth", [])) != 2474 or len(result.get("predicted", [])) != 2474):
        raise ValueError("结果不属于冻结的追加种子/历史五档验证协议")


def summarize(output, four_results, fixed_results):
    """分开列原三个、新三个、全部六个种子；按相同种子计算模型配对差。

    不按成绩筛种子，不将周期当作独立重复实验。目标Dev仅用于事后统计。
    保存每个输入结果文件SHA、全部类别指标及十对模型的逐种子差值。
    """
    p = config()
    conditions = ["clean" if k == "clean" else f"snr{k}" for k in p["target_conditions"]]
    cached, rows, classes, gates, hashes = {}, [], [], [], {}
    truth = None
    for method, seed in itertools.product(METHODS, ALL_SEEDS):
        root, expected_protocol = result_root(method, seed, output, four_results, fixed_results)
        path = root / f"models/{method}/seed{seed}/run.json"
        run = read(path)
        if (run["method"] != method or run["seed"] != seed or run["protocol"]["protocol"] != expected_protocol or
                any(run["protocol"][k] != p[k] for k in FAIR_KEYS) or
                run["target_used_for_training_or_selection"] is not False or run["final_holdout_evaluated"] is not False):
            raise ValueError("模型训练或选模配置与冻结协议不一致")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for condition in conditions:
            path = root / f"evaluations/{method}/seed{seed}/target_dev_{condition}.json"
            result = read(path)
            validate_result(result, method, seed, condition, expected_protocol)
            if result["selected_source_epoch"] != run["selected"]["epoch"]:
                raise ValueError("评估轮次与源域最佳权重不一致")
            if truth is None:
                truth = result["truth"]
            if result["truth"] != truth:
                raise ValueError("模型/条件的周期标签顺序不一致")
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            cached[method, seed, condition] = result
            base = {"method": method, "seed": seed, "condition": condition,
                    "seed_group": "added" if seed in NEW_SEEDS else "original"}
            metric = result["metrics"]
            rows.append({**base, "selected_source_epoch": result["selected_source_epoch"],
                         "macro_f1_percent": 100*metric["macro_f1"], "accuracy_percent": 100*metric["accuracy"],
                         "fault_as_healthy": metric["missed_fault_as_normal"], "healthy_as_fault": metric["normal_false_alarms"]})
            for label, name in enumerate(CLASS_NAMES):
                classes.append({**base, "label": label, "class_name": name,
                                "f1_percent": 100*metric["per_class_f1"][label],
                                "recall_percent": 100*metric["per_class_recall"][label],
                                "fault_as_healthy": metric["confusion_matrix"][label][0] if label else ""})
            if method == "nahsf":
                gate = result["gate"]
                gates.append({**base, **{k: gate[k] for k in ("mean", "std", "min", "max", "p10", "p50", "p90")}})
    seed_groups = {"original_three": OLD_SEEDS, "added_three": NEW_SEEDS, "combined_six": ALL_SEEDS}
    aggregates, pairs, paired_stats = [], [], []
    for group, seeds in seed_groups.items():
        for method, condition in itertools.product(METHODS, conditions):
            row = {"group": group, "seeds": list(seeds), "method": method, "condition": condition}
            for key in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
                row[key + "_percent"] = statistics([100*cached[method, s, condition]["metrics"][key] for s in seeds])
            aggregates.append(row)
        for left, right in itertools.combinations(METHODS, 2):
            for condition in conditions:
                paired_stats.append({"group": group, "seeds": list(seeds), "left": left, "right": right,
                    "condition": condition, "macro_f1_delta_pp": statistics([
                        100*(cached[left, s, condition]["metrics"]["macro_f1"]-
                             cached[right, s, condition]["metrics"]["macro_f1"]) for s in seeds])})
    for (left, right), seed, condition in itertools.product(itertools.combinations(METHODS, 2), ALL_SEEDS, conditions):
        a, b = cached[left, seed, condition], cached[right, seed, condition]
        ap, bp, y = np.asarray(a["predicted"]), np.asarray(b["predicted"]), np.asarray(truth)
        pairs.append({"left": left, "right": right, "seed": seed, "condition": condition,
                      "seed_group": "added" if seed in NEW_SEEDS else "original",
                      "macro_f1_delta_pp": 100*(a["metrics"]["macro_f1"]-b["metrics"]["macro_f1"]),
                      "accuracy_delta_pp": 100*(a["metrics"]["accuracy"]-b["metrics"]["accuracy"]),
                      "left_only_correct": int(((ap == y) & (bp != y)).sum()),
                      "right_only_correct": int(((bp == y) & (ap != y)).sum()),
                      "prediction_disagreements": int((ap != bp).sum())})
    write(output / "SUMMARY.json", {"protocol": p, "aggregates": aggregates, "paired_methods": paired_stats,
          "std_ddof": 1, "input_results_sha256": hashes,
          "historical_four_results": str(four_results), "historical_fixed_results": str(fixed_results)})
    csv_write(output / "PER_SEED_RESULTS.csv", rows)
    csv_write(output / "PER_CLASS_PER_SEED.csv", classes)
    csv_write(output / "PAIRED_PER_SEED.csv", pairs)
    csv_write(output / "NAHSF_GATE_PER_SEED.csv", gates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=NEW_SEEDS, default=list(NEW_SEEDS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strong-data-root", type=Path)
    parser.add_argument("--snr-root", type=Path)
    parser.add_argument("--four-results", type=Path)
    parser.add_argument("--fixed-results", type=Path)
    args = parser.parse_args()
    if args.stage in ("train", "evaluate") and args.strong_data_root is None:
        parser.error("需要已核验的0/5 dB验证/测试数据目录")
    if args.stage == "evaluate" and args.snr_root is None:
        parser.error("需要原10–40 dB实测数据目录")
    if args.stage == "summarize":
        if args.four_results is None or args.fixed_results is None:
            parser.error("需要最近两次实验的冻结历史结果目录")
        summarize(args.output, args.four_results, args.fixed_results)
    else:
        options = dict(protocol=config(), model_factory=make_model, version=VERSION)
        for method, seed in itertools.product(args.methods, args.seeds):
            if args.stage == "train":
                common.train(method, seed, args.output, args.strong_data_root, **options)
            else:
                common.evaluate(method, seed, args.output, args.snr_root, args.strong_data_root, **options)
