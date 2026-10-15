"""固定原V4的12份权重，在同一88个实测编号上评估10/5/0 dB。

独立入口只做推理：模型、切周期、滤波和归一化调用原运行目录的冻结代码；
10 dB重新推理并逐条核对原预测，5/0 dB读取已经生成的MAT。没有训练入口。
--v4-run-root 指原V4运行目录，--strong-data-root 指强噪声数据目录，
--output 必须是不存在的新目录，避免覆盖原结果或反复选择权重。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

METHODS = ("lowpass3", "lowpass4", "static4", "hpasf")
SEEDS = (42, 43, 44)
LEVELS = (10, 5, 0)
CLASS_NAMES = ("Healthy", "OUS(A)", "OLS(A)", "OP(A)", "OUS(A)&OUS(B)", "OLS(A)&OP(C)")


def read(path):
    """读取显式指定的配置或结果文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    """保存指标、预测标签和文件指纹，不导出原始电流或权重。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    """按文件字节核对旧权重、输入及代码是否发生变化。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_ids(split, metadata):
    """只接受原88个编号和2474个周期，拒绝保留集或新切分混入。"""
    ids = sorted({row["record_id"] for row in metadata})
    if (len(metadata) != 2474 or len(ids) != 88 or
            set(ids) != set(split["target_dev_records"]) or
            set(ids) & set(split["target_holdout_records"])):
        raise ValueError("必须使用原88个实测编号、2474个周期；不得包含最终保留集")
    return ids


def stats(values):
    """保留全部三个种子的数值；标准差为样本标准差，ddof=1。"""
    if len(values) != 3:
        raise ValueError("每组必须包含42/43/44三个种子")
    return {"mean": statistics.mean(values), "std": statistics.stdev(values), "values": values}


def summarize(evaluations):
    """在同一周期标签上汇总成绩、六类指标、漏报数量及逐种子配对差。

    退化值定义为10 dB减去5/0 dB；方法差定义为左侧方法减去右侧方法。
    先对同一种子求差，再求均值和标准差，不能用两组标准差相减。
    """
    cache = {(r["method"], r["seed"], r["snr_db"]): r for r in evaluations}
    expected = set(itertools.product(METHODS, SEEDS, LEVELS))
    if len(evaluations) != 36 or set(cache) != expected:
        raise ValueError("需要四方法×三种子×三档噪声的完整36组结果")
    truth = evaluations[0]["truth"]
    if any(r["truth"] != truth for r in evaluations):
        raise ValueError("各方法/噪声条件必须使用同一批周期及真实标签")
    aggregates, classes, drops, pairs = [], [], [], []
    for method, level in itertools.product(METHODS, LEVELS):
        scores = [cache[method, seed, level]["metrics"] for seed in SEEDS]
        row = {"method": method, "snr_db": level}
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
            row[key + "_percent"] = stats([100 * r[key] for r in scores])
        for key in ("missed_fault_as_normal", "normal_false_alarms"):
            row[key] = stats([r[key] for r in scores])
        aggregates.append(row)
        for label, name in enumerate(CLASS_NAMES):
            classes.append({"method": method, "snr_db": level, "label": label, "class_name": name,
                            "support": sum(truth_label == label for truth_label in truth),
                            "f1_percent": stats([100*r["per_class_f1"][label] for r in scores]),
                            "recall_percent": stats([100*r["per_class_recall"][label] for r in scores]),
                            "fault_as_healthy": (stats([r["confusion_matrix"][label][0] for r in scores])
                                                 if label > 0 else None)})
        if level != 10:
            row = {"method": method, "snr_db": level}
            for key in ("macro_f1", "accuracy"):
                row[key + "_drop_pp"] = stats([
                    100*(cache[method, seed, 10]["metrics"][key] -
                         cache[method, seed, level]["metrics"][key]) for seed in SEEDS])
            drops.append(row)
    # 四种方法全部六个两两配对均记录，包含static4对照，不按成绩筛选。
    for right, left in itertools.combinations(METHODS, 2):
        for level in LEVELS:
            row = {"left": left, "right": right, "snr_db": level}
            for key in ("macro_f1", "accuracy"):
                delta = [100*(cache[left, seed, level]["metrics"][key] -
                              cache[right, seed, level]["metrics"][key]) for seed in SEEDS]
                row[key + "_delta_pp"] = stats(delta)
            pairs.append(row)
    return {"seeds": list(SEEDS), "std_ddof": 1, "aggregates": aggregates,
            "per_class": classes, "drop_from_10db": drops, "paired_methods": pairs}


def csv_write(path, rows):
    """保存便于检查和绘图的逐种子数值表。"""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_tables(output, evaluations, summary):
    """指标单位统一为百分数/百分点；数量保留为整数。"""
    rows, classes = [], []
    for r in evaluations:
        m = r["metrics"]
        base = {"method": r["method"], "seed": r["seed"], "snr_db": r["snr_db"]}
        rows.append({**base, "macro_f1_percent": 100*m["macro_f1"],
                     "accuracy_percent": 100*m["accuracy"],
                     "missed_fault_as_normal": m["missed_fault_as_normal"],
                     "normal_false_alarms": m["normal_false_alarms"]})
        for label, name in enumerate(CLASS_NAMES):
            classes.append({**base, "label": label, "class_name": name,
                            "support": sum(r["metrics"]["confusion_matrix"][label]),
                            "f1_percent": 100*m["per_class_f1"][label],
                            "recall_percent": 100*m["per_class_recall"][label],
                            "fault_as_healthy": m["confusion_matrix"][label][0] if label else ""})
    csv_write(output / "PER_SEED_RESULTS.csv", rows)
    csv_write(output / "PER_CLASS_PER_SEED.csv", classes)
    pairs = []
    for r in summary["paired_methods"]:
        for index, seed in enumerate(SEEDS):
            pairs.append({"left": r["left"], "right": r["right"], "snr_db": r["snr_db"],
                          "seed": seed, "macro_f1_delta_pp": r["macro_f1_delta_pp"]["values"][index],
                          "accuracy_delta_pp": r["accuracy_delta_pp"]["values"][index]})
    csv_write(output / "PAIRED_METHODS.csv", pairs)


def evaluate(args):
    """校验来源、一次切出三档相同周期、加载旧权重并纯推理。

    原V4代码目录位于sys.path首位，prepare.ROOT仍指向原冻结数据链接。
    因此不复制权重、不重写原脚本，也不改变原数据池或预处理实现。
    """
    started = time.perf_counter()
    run, strong, output = args.v4_run_root.resolve(), args.strong_data_root.resolve(), args.output.resolve()
    if output.exists() or output == run or run in output.parents or strong in output.parents:
        raise ValueError("结果必须保存到独立且不存在的新目录")
    deployment = read(run / "DEPLOYMENT_MANIFEST.json")
    old_check = read(run / "results/VERIFICATION.json")
    generated = read(strong / "generated/GENERATION_MANIFEST.json")
    if any(r["status"] != "passed" for r in (old_check, generated, read(strong / "VERIFICATION.json"))):
        raise ValueError("原V4或强噪声数据尚未通过校验")
    for relative, expected in deployment["code_sha256"].items():
        if sha(run / "code" / relative) != expected:
            raise ValueError(f"原冻结代码/划分已改变：{relative}")
    split = read(run / "code/configs/splits_gate0_v1.json")
    metadata_path = run / "code/outputs/gate0_v1/target_dev_clean_windows.json"
    metadata = read(metadata_path)
    ids = frozen_ids(split, metadata)
    if sha(run / "code/configs/splits_gate0_v1.json") != generated["split_sha256"]:
        raise ValueError("强噪声数据与原V4数据划分不一致")
    files = {r["file"]: r for r in generated["files"]}
    selected = []
    for level, rid in itertools.product((5, 0), ids):
        relative = f"target/SNR_{level}/{rid}.mat"
        r = files[relative]
        if r["partition"] != "dev" or sha(strong / "generated" / relative) != r["file_sha256"]:
            raise ValueError(f"实测文件来源或字节不一致：{relative}")
        selected.append(r)

    # 把旧结果、权重、冻结代码及本轮输入的指纹保存下来，结束后逐文件复核。
    protected = [p for p in (run / "results").rglob("*") if p.is_file()]
    protected += [run / "code" / p for p in deployment["code_sha256"]]
    protected += [metadata_path, strong / "generated/GENERATION_MANIFEST.json"]
    protected += [strong / "generated" / r["file"] for r in selected]
    protected += [run / "target_snr/SNR_10" / f"{rid}.mat" for rid in ids]
    protected += [run / "code/data/target_dev/clean" / f"{rid}.mat" for rid in ids]
    before = {str(p): sha(p) for p in protected}
    sys.path.insert(0, str(run / "code/scripts"))
    import numpy as np
    import torch
    import v2_experiment as v2
    from v1_noise_consistency import fix_randomness, metrics
    from v4_hpasf import HPASFCNN, RAW_POINTS
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("本轮约定在服务器GPU执行，请检查既有PyTorch环境")
    # 每档MAT只解析一次，所有方法和种子共享输入，不重复加噪。
    inputs = {}
    for level in LEVELS:
        root = run / "target_snr" if level == 10 else strong / "generated/target"
        raw, actual_meta = v2.target_dev_from_mat(level, root)
        if actual_meta != metadata:
            raise ValueError("周期边界与原实验不一致")
        inputs[level] = v2.raw_cycles(raw, metadata)
    input_seconds = time.perf_counter() - started
    y = np.asarray([r["label"] for r in metadata], dtype=np.int64)
    output.mkdir(parents=True, exist_ok=False)
    evaluations = []
    inference_started = time.perf_counter()
    for method, seed in itertools.product(METHODS, SEEDS):
        fix_randomness(seed)
        path = run / "results/models" / method / f"seed{seed}/best.pt"
        weight_sha = sha(path)
        if weight_sha != old_check["weights_sha256"][f"{method}/seed{seed}"]:
            raise ValueError(f"权重指纹不一致：{path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("code_family") != "HUST-V4-HPASF-v0" or
                checkpoint.get("source_only") is not True or checkpoint.get("mode") != method or
                checkpoint.get("seed") != seed or checkpoint.get("raw_points") != RAW_POINTS):
            raise ValueError("权重版本、模型、种子或输入长度不一致")
        model = HPASFCNN(method).to(device)
        model.load_state_dict(checkpoint["state"], strict=True)
        for level in LEVELS:
            with torch.inference_mode():
                prediction = v2.inference(model, inputs[level], device)
            score = metrics(y, prediction)
            if level == 10:
                original = read(run / "results/evaluations" / method / f"seed{seed}/target_dev_snr10.json")
                if (original["truth"] != y.tolist() or original["predicted"] != prediction.tolist() or
                        original["metrics"] != score or original["source_only_selection"] != checkpoint["selected"]):
                    raise ValueError(f"10 dB不能完全复现：{method}/seed{seed}")
            result = {"method": method, "seed": seed, "snr_db": level, "record_families": 88,
                      "windows": len(y), "metrics": score, "truth": y.tolist(),
                      "predicted": prediction.tolist(), "weights_sha256": weight_sha,
                      "selected_original_epoch": checkpoint["selected"]["epoch"],
                      "retrained": False, "final_holdout_evaluated": False}
            write(output / "evaluations" / method / f"seed{seed}/target_dev_snr{level}.json", result)
            evaluations.append(result)
            print(f"{method}/seed{seed} {level} dB F1={100*score['macro_f1']:.4f}% "
                  f"Accuracy={100*score['accuracy']:.4f}%", flush=True)
        # 包含BatchNorm缓冲区在内，确认推理没有修改任何权重或统计量。
        if model.training or any(not torch.equal(value.cpu(), checkpoint["state"][key])
                                 for key, value in model.state_dict().items()):
            raise RuntimeError("推理改变了模型状态")
    inference_seconds = time.perf_counter() - inference_started
    summary = summarize(evaluations)
    write(output / "SUMMARY.json", summary)
    export_tables(output, evaluations, summary)
    if any(sha(Path(path)) != expected for path, expected in before.items()):
        raise RuntimeError("原权重、结果、代码或输入文件被改写")
    write(output / "RUN_RECORD.json", {
        "status": "passed", "original_v4_code_commit": deployment["code_commit"],
        "evaluation_script_sha256": sha(Path(__file__)), "v4_run_root": str(run),
        "strong_data_root": str(strong), "methods": list(METHODS), "seeds": list(SEEDS),
        "conditions_db": list(LEVELS), "new_5_0_evaluations": 24, "original_10db_exact_replays": 12,
        "record_ids": ids, "class_names": list(CLASS_NAMES), "windows_per_condition": len(y),
        "class_support": np.bincount(y, minlength=6).tolist(), "retrained_models": 0,
        "optimizer_updates": 0, "state_dicts_unchanged": 12, "final_holdout_evaluated": False,
        "target_noise": "clean + (author10-clean)*10**((10-SNR)/20)",
        "noise_realizations_independent_across_snr": False, "environment_reused": True,
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "python_version": sys.version, "input_preparation_seconds": input_seconds,
        "inference_36_evaluations_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started, "selected_generated_files": selected,
        "protected_files_unchanged": before})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("v4-run-root", "strong-data-root", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    evaluate(parser.parse_args())
