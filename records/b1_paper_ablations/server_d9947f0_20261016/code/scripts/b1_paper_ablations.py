"""Independent Source-Cycle V2 experiment runner for A1–A5 B1 ablations.

Frozen copy of the verified benchmark training/evaluation protocol, except
the explicit five ablation model registry, 42/44 seeds and four target SNRs.
All artifacts go to a NEW root; baseline scripts/models remain unchanged.
Only simulation-source data are used for training and checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from hust_a1_b1_experiment import select_source_checkpoint
from b1_paper_ablation_models import (METHODS as ABLATION_METHODS, build_model as build_ablation_model,
    paired_loss as ablation_paired_loss, pretrain_restorer, calibrate_a2_from_source)
from hust_arch_suite_experiment import digest_parameters, write_csv
from source_cycle_split import source_views, source_provenance, DEFAULT_OUT as DATA_ROOT
from v1_noise_consistency import fix_randomness, metrics
import v2_experiment as data

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/b1_paper_ablations_v1.json"
PROTOCOL = "hust-cycle-v2-b1-paper-ablations-42-44-v1"
METHODS = ABLATION_METHODS
SEEDS = (42, 44)
CONDITIONS = (5, 0, -5, -10)
OUT = Path("outputs/b1_paper_ablations_v1")


def read(path: Path):
    """读取本轮冻结JSON，不改写历史配置或结果。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    """固定UTF-8、键排序与换行，保证两端记录可逐字节核查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def configuration() -> dict:
    """本协议独立记录六种子，不修改仍用于复现的旧三种子配置。"""
    config = read(CONFIG)
    if config["protocol"] != PROTOCOL or config["methods"] != list(METHODS) or config["seeds"] != list(SEEDS):
        raise ValueError("本轮模型、种子或协议改变")
    return config


def build_model(method: str) -> torch.nn.Module:
    """All A1..A5 variants use source-controlled isolated constructors."""
    return build_ablation_model(method)


def objective(method, model, clean, noisy, labels, config):
    return ablation_paired_loss(
        method, model, clean, noisy, labels,
        lambda_restore=config["b1_restore_weight"])


def logits(model, x):
    """Xu返回重构与特征字典，其余模型返回分类张量；仅统一读取接口。"""
    output = model(x)
    return output["logits"] if isinstance(output, dict) else output


@torch.no_grad()
def predict(model, x, device, labels=None, batch=128):
    """一次推理同时计算预测与按样本加权CE，避免重复执行验证前向。"""
    model.eval()
    predictions, ce_sum = [], 0.0
    for start in range(0, len(x), batch):
        scores = logits(model, torch.from_numpy(x[start:start + batch]).to(device))
        predictions.append(scores.argmax(1).cpu().numpy())
        if labels is not None:
            target = torch.from_numpy(labels[start:start + batch]).to(device)
            ce_sum += float(F.cross_entropy(scores, target, reduction="sum").cpu())
    return np.concatenate(predictions), ce_sum / len(x) if labels is not None else None


def snapshot(model):
    """复制参数及BN缓冲区，后续训练不能改变已记录轮次。"""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checked_source(root, config):
    """只接受已经跨平台核验的新数据版本，拒绝回退到738/302旧划分。"""
    provenance = source_provenance(root)
    if provenance["manifest_sha256"] != config["source_split_manifest_sha256"]:
        raise ValueError("仿真划分指纹与本轮配置不符")
    views = source_views(root)
    if (len(views[0][2]), len(views[1][2])) != (config["source_train_cycles"], config["source_val_cycles"]):
        raise ValueError("训练/验证数量改变")
    return views, provenance


def train(method: str, seed: int, out: Path, source_root: Path):
    """一组独立源域训练；先完成50轮，再按源验证选中best，不接触目标数据。"""
    config = configuration()
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("模型或种子不属于本轮")
    folder = out / "models" / method / f"seed{seed}"
    if folder.exists():
        raise FileExistsError(folder)
    fix_randomness(seed)
    views, provenance = checked_source(source_root, config)
    (raw, metadata, y, clean), (_, _, vy, val_clean, val_noise) = views
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(method).to(device)
    # A2 is the ONLY variant with a frozen source-derived noise buffer.
    # Calibrate before optimizer construction and BEFORE restoration pretraining.
    calibrate_a2_from_source(
        model, raw, metadata, seed, config["train_noise_db"])
    initial = digest_parameters(model)
    pretraining = pretrain_restorer(
        method, model, raw, metadata, clean, seed, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, states, first_noise_sha = [], {}, None
    for epoch in range(1, config["epochs"] + 1):
        noisy = data.fresh_noisy_source(raw, metadata, seed, epoch, noise_levels=config["train_noise_db"])
        if epoch == 1:
            first_noise_sha = hashlib.sha256(noisy.tobytes()).hexdigest()
        order = np.random.default_rng(seed + 901 + epoch).permutation(len(y))
        model.train()
        losses, weighted_loss, components = [], 0.0, {}
        for start in range(0, len(order), config["batch_size"]):
            selected = order[start:start + config["batch_size"]]
            cx = torch.from_numpy(clean[selected]).to(device)
            nx = torch.from_numpy(noisy[selected]).to(device)
            labels = torch.from_numpy(y[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(method, model, cx, nx, labels, config)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"损失非有限：{method}/{seed}/{epoch}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError("梯度非有限")
            optimizer.step()
            scalar = float(loss.detach().cpu())
            losses.append(scalar)
            weighted_loss += scalar * len(selected)
            for key, value in terms.items():
                components.setdefault(key, []).append(float(value.detach().cpu()))
        clean_pred, clean_ce = predict(model, val_clean, device, vy)
        val_scores, val_ce = {}, {}
        for snr, values in val_noise.items():
            predicted, ce = predict(model, values, device, vy)
            val_scores[str(snr)] = metrics(vy, predicted)["macro_f1"]
            val_ce[str(snr)] = ce
        row = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
               "train_loss_sample_weighted": weighted_loss / len(y),
               "train_components": {key: float(np.mean(value)) for key, value in components.items()},
               "optimizer_steps": len(losses), "clean_f1": metrics(vy, clean_pred)["macro_f1"],
               "clean_ce": clean_ce, "noisy_f1": val_scores, "noisy_ce": val_ce,
               "noise_mean_f1": float(np.mean(list(val_scores.values()))),
               "noise_mean_ce": float(np.mean(list(val_ce.values())))}
        history.append(row)
        states[epoch] = snapshot(model)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{method} seed{seed} epoch{epoch}/50 source F1={row['noise_mean_f1']:.6f} CE={row['noise_mean_ce']:.6f}", flush=True)
    chosen = select_source_checkpoint(history)
    folder.mkdir(parents=True, exist_ok=False)
    common = {"method": method, "seed": seed, "protocol": PROTOCOL, "config": config,
              "source_split": provenance, "source_only": True, "final_holdout_evaluated": False}
    torch.save({**common, "state": states[chosen["epoch"]], "selected": chosen}, folder / "best.pt")
    torch.save({**common, "state": states[config["epochs"]], "epoch": config["epochs"]}, folder / "last.pt")
    write(folder / "history.json", history)
    write(folder / "run.json", {**common, "selected": chosen, "initial_parameters_sha256": initial,
          "parameters": sum(p.numel() for p in model.parameters()), "pretrain_history": pretraining,
          "source_train_cycles": len(y), "source_val_cycles": len(vy),
          "first_training_noise_sha256": first_noise_sha,
          "target_used_for_training_or_selection": False})


def target_cycles(snr, snr_root, strong_root, extreme_root):
    """所有条件均通过原88编号读取器，不扫描或读取最终保留集。"""
    root = strong_root if snr in (0, 5) else extreme_root if snr in (-5, -10) else snr_root
    raw, metadata = data.target_dev_from_mat(snr, root)
    if len(raw) != 88 or len(metadata) != 2474:
        raise ValueError("实测Dev队列改变")
    return data.raw_cycles(raw, metadata), metadata


def evaluate(out: Path, methods, seeds, conditions, source_root, snr_root, strong_root, extreme_root):
    """Load frozen source-selected best; use the same 88 measured Dev families."""
    config = configuration()
    provenance = source_provenance(source_root)
    if provenance["manifest_sha256"] != config["source_split_manifest_sha256"]:
        raise ValueError("评价所关联的源划分改变")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for method in methods:
        for seed in seeds:
            folder = out / "models" / method / f"seed{seed}"
            saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            run = read(folder / "run.json")
            for key, expected in (("method", method), ("seed", seed), ("protocol", PROTOCOL),
                                  ("config", config), ("source_split", provenance)):
                if saved[key] != expected or run[key] != expected:
                    raise ValueError(f"权重来源不一致：{key}")
            if run["target_used_for_training_or_selection"] or saved["final_holdout_evaluated"]:
                raise ValueError("目标数据使用标记异常")
            fix_randomness(seed)
            model = build_model(method).to(device).eval()
            model.load_state_dict(saved["state"], strict=True)
            models[method, seed] = (model, saved["selected"])
    for snr in conditions:
        name = "clean" if snr is None else f"snr{snr}"
        x, metadata = target_cycles(snr, snr_root, strong_root, extreme_root)
        truth = np.asarray([row["label"] for row in metadata], dtype=np.int64)
        for (method, seed), (model, chosen) in models.items():
            path = out / f"evaluations/{method}/seed{seed}/target_dev_{name}.json"
            if path.exists():
                raise FileExistsError(path)
            predicted, _ = predict(model, x, device)
            score = metrics(truth, predicted)
            write(path, {"method": method, "seed": seed, "protocol": PROTOCOL, "condition": name,
                  "source_split": provenance, "record_families": 88, "windows": len(truth),
                  "truth": truth.tolist(), "predicted": predicted.tolist(), "metrics": score,
                  "source_only_selection": chosen, "target_used_for_training_or_selection": False,
                  "final_holdout_evaluated": False})
            print(f"DEV {method} seed{seed} {name} Macro-F1={100*score['macro_f1']:.3f}%", flush=True)


def summarize(out: Path):
    """Require all five variants x 2 seeds x 4 SNR, sample SD (ddof=1)."""
    config = configuration()
    per_seed, per_class, summary = [], [], []
    common_truth = None
    common_split = None
    for method in METHODS:
        for seed in SEEDS:
            for snr in CONDITIONS:
                name = "clean" if snr is None else f"snr{snr}"
                obj = read(out / f"evaluations/{method}/seed{seed}/target_dev_{name}.json")
                if (obj["method"], obj["seed"], obj["condition"], obj["protocol"]) != (method, seed, name, PROTOCOL):
                    raise ValueError("评价标识不一致")
                if obj["final_holdout_evaluated"] or obj["record_families"] != 88 or obj["windows"] != 2474:
                    raise ValueError("评价队列或保留集标记异常")
                if common_truth is None:
                    common_truth, common_split = obj["truth"], obj["source_split"]
                if obj["truth"] != common_truth or obj["source_split"] != common_split:
                    raise ValueError("不能汇总不同数据划分或不同目标队列")
                score = obj["metrics"]
                per_seed.append({"method": method, "seed": seed, "condition": name,
                    "macro_f1_percent": 100 * score["macro_f1"], "accuracy_percent": 100 * score["accuracy"],
                    "fault_as_healthy": score["missed_fault_as_normal"], "healthy_as_fault": score["normal_false_alarms"]})
                for label, (f1, recall) in enumerate(zip(score["per_class_f1"], score["per_class_recall"])):
                    per_class.append({"method": method, "seed": seed, "condition": name, "label": label,
                                      "f1_percent": 100 * f1, "recall_percent": 100 * recall})
    for method in METHODS:
        for snr in CONDITIONS:
            name = "clean" if snr is None else f"snr{snr}"
            rows = [r for r in per_seed if r["method"] == method and r["condition"] == name]
            if len(rows) != len(SEEDS):
                raise ValueError("种子结果不完整")
            row = {"method": method, "condition": name, "seeds": len(rows)}
            for key in ("macro_f1", "accuracy"):
                values = np.asarray([r[key + "_percent"] for r in rows])
                row[key + "_mean"] = float(values.mean())
                row[key + "_sd"] = float(values.std(ddof=1))
            summary.append(row)
    reports = out / "reports"
    write_csv(reports / "PER_SEED_RESULTS.csv", per_seed)
    write_csv(reports / "PER_CLASS_PER_SEED.csv", per_class)
    write_csv(reports / "SUMMARY.csv", summary)
    write(reports / "SUMMARY.json", {"protocol": PROTOCOL, "config": config,
          "source_split": common_split, "summary": summary, "final_holdout_evaluated": False})


def main():
    """Only the five frozen ablations and 42/44 seeds are accepted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--snr-root", type=Path)
    parser.add_argument("--strong-root", type=Path)
    parser.add_argument("--extreme-root", type=Path)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("模型或种子不能重复")
    if args.stage == "train":
        for method in args.methods:
            for seed in args.seeds:
                train(method, seed, args.out, args.source_root)
    elif args.stage == "evaluate":
        if not all((args.strong_root, args.extreme_root)):
            parser.error("四档评价需要 --strong-root 和 --extreme-root")
        evaluate(args.out, args.methods, args.seeds, CONDITIONS, args.source_root,
                 args.snr_root, args.strong_root, args.extreme_root)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
