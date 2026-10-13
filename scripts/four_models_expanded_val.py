"""四模型统一实验：训练10–40 dB，验证0/5/10/20/30 dB。

四模型为无滤波A′、三阶低通、四阶低通和V5 NA-HSF。共用同一训练循环、
原始数据池、噪声随机流和损失；仅验证档位及选模平均值范围发生改变。
运行结果必须放在独立新目录，旧V2/V4/V5入口和模型实现不修改。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import v2_experiment as data
from v2_paifilter import HarmonicCNN, RAW_POINTS, paired_cross_entropy
from v4_hpasf import HPASFCNN
from v5_nahsf import NAHSFCNN
from v5_experiment import predict_with_gate, gate_summary
from v1_noise_consistency import NOISE_LEVELS, fix_randomness, metrics, score_checkpoints
from evaluate_v4_strong_snr import read, write, csv_write, stats, frozen_ids, CLASS_NAMES

CONFIG = Path(__file__).resolve().parents[1] / "configs/four_models_expanded_val_v1.json"
METHODS = ("identity", "lowpass3", "lowpass4", "nahsf")
VERSION = "HUST-four-models-expanded-val-v1"


def config():
    """校验冻结协议，防止把上轮扩大训练噪声的配置混入本轮。"""
    value = read(CONFIG)
    expected = {"methods": list(METHODS), "seeds": [42, 43, 44], "epochs": 50,
                "batch_size": 32, "learning_rate": .001,
                "training_noise_levels_db": list(NOISE_LEVELS),
                "validation_noise_levels_db": [0, 5, 10, 20, 30],
                "target_conditions": ["clean", 10, 15, 20, 25, 30, 35, 40, 5, 0]}
    if any(value[k] != v for k, v in expected.items()):
        raise ValueError("配置与四模型扩大验证协议不一致")
    return value


def make_model(method):
    """只构造请求的模型，避免额外模型消耗随机数而改变CNN初值。

    identity保留MD归一化和128点输入，只去掉频域滤波。其他三种直接调用
    已核验的原模型类，不增加参数、不替换Backbone。
    """
    if method == "identity":
        return HarmonicCNN("identity")
    if method in ("lowpass3", "lowpass4"):
        return HPASFCNN(method)
    if method == "nahsf":
        return NAHSFCNN()
    raise ValueError(f"未知模型：{method}")


def parameter_counts(model):
    """分开记录同一CNN和滤波/门控的参数数目。"""
    cnn = sum(p.numel() for p in model.backbone.parameters())
    spectral = sum(p.numel() for p in model.shaping.parameters())
    return {"cnn": cnn, "spectral": spectral, "total": cnn + spectral}


def select_original_reference(history):
    """仅为数值核查重算旧三档选模轮次，不参与本轮测试或最终选模。"""
    original = [{**r, "noise_mean_f1": r["original_three_snr_mean_f1"]} for r in history]
    return score_checkpoints(original)


def train(method, seed, output, strong_root):
    """所有模型共用原训练随机流，新增验证不更新BatchNorm或任何权重。

    best.pt按五档验证保存；old_selection_reference.pt只是同次训练的审计
    快照，用于与旧实验的权重逐张量比较，不额外训练，也不用于目标评估。
    """
    protocol = config()
    folder = output / f"models/{method}/seed{seed}"
    if folder.exists():
        raise FileExistsError(f"拒绝覆盖模型：{folder}")
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, tm, y = data.source_data("train")
    vr, vm, vy = data.source_data("val")
    clean, clean_val = data.raw_cycles(tr, tm), data.raw_cycles(vr, vm)
    val = data.fixed_noisy_validation(vr, vm, noise_levels=tuple(protocol["validation_noise_levels_db"]),
                                     materialized_root=strong_root / "generated/source")
    model = make_model(method).to(device)
    initial_sha = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    history, states = [], {}
    for epoch in range(1, 51):
        noisy = data.fresh_noisy_source(tr, tm, seed, epoch)  # 默认原10–40 dB，明确不加入0/5
        permutation = np.random.default_rng(seed + 901 + epoch).permutation(len(y))
        model.train()
        losses, gate_gradient = [], []
        for start in range(0, len(y), 32):
            indices = permutation[start:start + 32]
            cx, nx = torch.from_numpy(clean[indices]).to(device), torch.from_numpy(noisy[indices]).to(device)
            labels = torch.from_numpy(y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = paired_cross_entropy(model, cx, nx, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"训练loss非有限值：{method}/{seed}/{epoch}")
            loss.backward()
            if method == "nahsf":
                gradient = model.shaping.gate[-1].bias.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("NA-HSF门控梯度缺失或非有限值")
                gate_gradient.append(float(gradient.detach().abs().mean().cpu()))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        clean_f1 = metrics(vy, data.inference(model, clean_val, device))["macro_f1"]
        noisy_f1 = {str(level): metrics(vy, data.inference(model, x, device))["macro_f1"] for level, x in val.items()}
        row = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)), "clean_f1": clean_f1,
               "noisy_f1": noisy_f1, "noise_mean_f1": float(np.mean(list(noisy_f1.values()))),
               "original_three_snr_mean_f1": float(np.mean([noisy_f1[str(k)] for k in (10, 20, 30)]))}
        if method == "nahsf":
            row.update(gate_bias_grad_abs_mean=float(np.mean(gate_gradient)),
                       gate_bias_value=float(model.shaping.gate[-1].bias.detach().cpu().item()))
        history.append(row)
        states[epoch] = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"{method}/seed{seed} epoch{epoch}/50 Clean={clean_f1:.4f} "
                  f"Val5={row['noise_mean_f1']:.4f} Val3={row['original_three_snr_mean_f1']:.4f}", flush=True)
    chosen, reference = score_checkpoints(history), select_original_reference(history)
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint = {"state": states[chosen["epoch"]], "method": method, "seed": seed, "selected": chosen,
                  "source_only": True, "raw_points": RAW_POINTS, "code_family": VERSION,
                  "protocol": protocol["protocol"], "train_noise_levels_db": protocol["training_noise_levels_db"],
                  "val_noise_levels_db": protocol["validation_noise_levels_db"], "final_holdout_evaluated": False}
    torch.save(checkpoint, folder / "best.pt")
    torch.save({"state": states[reference["epoch"]], "epoch": reference["epoch"],
                "purpose": "old three-SNR selection numerical audit only"}, folder / "old_selection_reference.pt")
    write(folder / "history.json", history)
    write(folder / "run.json", {"method": method, "seed": seed, "protocol": protocol,
                                "initial_cnn_sha256": initial_sha, "parameter_counts": parameter_counts(model),
                                "selected": chosen, "old_three_snr_selected": reference,
                                "target_used_for_training_or_selection": False, "final_holdout_evaluated": False})


def check_checkpoint(checkpoint, method, seed):
    """禁止拿旧权重或扩大训练噪声的权重冒充本轮重新选模结果。"""
    protocol = config()
    expected = {"method": method, "seed": seed, "code_family": VERSION, "source_only": True,
                "raw_points": RAW_POINTS, "train_noise_levels_db": protocol["training_noise_levels_db"],
                "val_noise_levels_db": protocol["validation_noise_levels_db"], "protocol": protocol["protocol"]}
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError("权重与本轮四模型扩大验证协议不一致")


def evaluate(method, seed, output, original_root, strong_root):
    """同一88个编号、2474周期，纯推理；所有条件只用五档验证选中的best.pt。"""
    protocol = config()
    fix_randomness(seed)
    checkpoint = torch.load(output / f"models/{method}/seed{seed}/best.pt", map_location="cpu", weights_only=True)
    check_checkpoint(checkpoint, method, seed)
    model = make_model(method).cuda()
    model.load_state_dict(checkpoint["state"], strict=True)
    from prepare import ROOT, OUT
    split, metadata = read(ROOT / "configs/splits_gate0_v1.json"), read(OUT / "target_dev_clean_windows.json")
    frozen_ids(split, metadata)
    for level in protocol["target_conditions"]:
        snr = None if level == "clean" else level
        name = "clean" if snr is None else f"snr{snr}"
        path = output / f"evaluations/{method}/seed{seed}/target_dev_{name}.json"
        if path.exists():
            raise FileExistsError(path)
        root = strong_root / "generated/target" if snr in (0, 5) else original_root
        raw, actual_meta = data.target_dev_from_mat(snr, root)
        if actual_meta != metadata:
            raise ValueError("实测周期顺序发生改变")
        truth = np.array([r["label"] for r in metadata], dtype=np.int64)
        x = data.raw_cycles(raw, metadata)
        gate = None
        if method == "nahsf":
            prediction, alpha, descriptor = predict_with_gate(model, x, torch.device("cuda"))
            gate = gate_summary(alpha, descriptor, truth)
        else:
            prediction = data.inference(model, x, torch.device("cuda"))
        result = {"method": method, "seed": seed, "condition": name, "record_families": 88, "windows": len(truth),
                  "metrics": metrics(truth, prediction), "truth": truth.tolist(), "predicted": prediction.tolist(),
                  "gate": gate, "protocol": protocol["protocol"], "selected_source_epoch": checkpoint["selected"]["epoch"],
                  "target_used_for_training_or_selection": False, "final_holdout_evaluated": False}
        write(path, result)
        print(f"{method}/seed{seed} {name} Macro-F1={100*result['metrics']['macro_f1']:.4f}%", flush=True)


def summarize(output):
    """三个种子统计和六个模型配对；另保存六类指标，保留全部结果。"""
    protocol = config()
    conditions = ["clean" if x == "clean" else f"snr{x}" for x in protocol["target_conditions"]]
    cached, rows, classes = {}, [], []
    truth = None
    for method, seed, condition in itertools.product(METHODS, protocol["seeds"], conditions):
        result = read(output / f"evaluations/{method}/seed{seed}/target_dev_{condition}.json")
        if (result["method"] != method or result["seed"] != seed or result["condition"] != condition or
                result["protocol"] != protocol["protocol"] or result["windows"] != 2474 or
                result["record_families"] != 88 or result["final_holdout_evaluated"] is not False):
            raise ValueError("结果来源不一致")
        if truth is None:
            truth = result["truth"]
        if result["truth"] != truth:
            raise ValueError("不同模型或条件的周期标签顺序不一致")
        metric = result["metrics"]
        cached[method, seed, condition] = metric
        base = {"method": method, "seed": seed, "condition": condition}
        rows.append({**base, "macro_f1_percent": 100*metric["macro_f1"], "accuracy_percent": 100*metric["accuracy"],
                     "fault_as_healthy": metric["missed_fault_as_normal"], "healthy_as_fault": metric["normal_false_alarms"]})
        for label, name in enumerate(CLASS_NAMES):
            classes.append({**base, "label": label, "class_name": name, "f1_percent": 100*metric["per_class_f1"][label],
                            "recall_percent": 100*metric["per_class_recall"][label],
                            "fault_as_healthy": metric["confusion_matrix"][label][0] if label else ""})
    aggregates, pairs = [], []
    for method, condition in itertools.product(METHODS, conditions):
        row = {"method": method, "condition": condition}
        for key in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
            row[key + "_percent"] = stats([100*cached[method, seed, condition][key] for seed in protocol["seeds"]])
        aggregates.append(row)
    for right, left in itertools.combinations(METHODS, 2):
        for condition in conditions:
            pairs.append({"left": left, "right": right, "condition": condition,
                          "macro_f1_delta_pp": stats([100*(cached[left, seed, condition]["macro_f1"]-
                                                           cached[right, seed, condition]["macro_f1"]) for seed in protocol["seeds"]])})
    write(output / "SUMMARY.json", {"protocol": protocol, "aggregates": aggregates, "paired_methods": pairs, "std_ddof": 1})
    csv_write(output / "PER_SEED_RESULTS.csv", rows)
    csv_write(output / "PER_CLASS_PER_SEED.csv", classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strong-data-root", type=Path)
    parser.add_argument("--snr-root", type=Path)
    args = parser.parse_args()
    if args.stage in ("train", "evaluate") and args.strong_data_root is None:
        parser.error("需要固定0/5 dB验证/测试文件的目录")
    if args.stage == "evaluate" and args.snr_root is None:
        parser.error("需要原10–40 dB实测数据目录")
    if args.stage == "summarize":
        summarize(args.output)
    else:
        for mode, seed in itertools.product(args.methods, config()["seeds"]):
            if args.stage == "train":
                train(mode, seed, args.output, args.strong_data_root)
            else:
                evaluate(mode, seed, args.output, args.snr_root, args.strong_data_root)
