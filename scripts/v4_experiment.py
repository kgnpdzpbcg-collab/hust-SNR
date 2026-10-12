"""Four-way V4 HPASF source-only comparison on the FROZEN HUST-SNR protocol.

Methods: lowpass3 / lowpass4 / static4 / hpasf; CNN architecture, noise RNG,
source splits, CE objective, training epochs and checkpoint selection are
identical to the already verified V2 runner. Target: only 88 Dev record
families; 347 Final-Holdout records are NEVER loaded by this script.

    python scripts/v4_experiment.py smoke
    python scripts/v4_experiment.py train
    python scripts/v4_experiment.py evaluate --snr-root PATH_TO_SNR_PARENT
    python scripts/v4_experiment.py summarize
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import v2_experiment as v2
from v1_noise_consistency import (NOISE_LEVELS, VAL_NOISE_LEVELS,
                                  fix_randomness, metrics, score_checkpoints)
from v2_paifilter import RAW_POINTS, paired_cross_entropy
from v4_hpasf import HPASFCNN, METHODS, INITIAL_GAIN4, counts_by_module

DEFAULT_OUTPUT = Path("outputs/v4_hpasf")
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_CONDITIONS = [None, 10, 15, 20, 25, 30, 35, 40]
SOURCE_SELECTION_RULE = "source 10/20/30 noisy-val mean Macro-F1; Clean within 1pp of peak"
VERSION = "HUST-V4-HPASF-v0"


def train(mode: str, seed: int, output: Path, epochs: int = 50,
          batch_size: int = 32, lr: float = 0.001, *,
          train_noise_levels=NOISE_LEVELS, val_noise_levels=VAL_NOISE_LEVELS,
          validation_mat_root: Path | None = None,
          train_protocol: str = "V2_M1_paired_source_CE") -> None:
    """共用原训练循环；独立强噪声入口显式传入档位和固定验证MAT路径。

    默认参数保留原实验。模型、损失、优化器、50轮预算和Clean约束不变；
    新旧协议由权重及run.json的train_protocol/噪声档位区分。
    """
    if mode not in METHODS or epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid training configuration")
    dest = output / "models" / mode / f"seed{seed}"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {dest}")
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_raw, tr_meta, tr_y = v2.source_data("train")
    va_raw, va_meta, va_y = v2.source_data("val")
    clean_train = v2.raw_cycles(tr_raw, tr_meta)
    clean_val = v2.raw_cycles(va_raw, va_meta)
    fixed_noisy_val = v2.fixed_noisy_validation(
        va_raw, va_meta, noise_levels=val_noise_levels, materialized_root=validation_mat_root)
    selection_rule = ("source " + "/".join(map(str, val_noise_levels)) +
                      " noisy-val mean Macro-F1; Clean within 1pp of peak")

    model = HPASFCNN(mode).to(device)
    initial_cnn_sha = hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    history: list[dict] = []
    states: dict[int, dict] = {}
    for epoch in range(1, epochs + 1):
        noisy_train = v2.fresh_noisy_source(tr_raw, tr_meta, seed, epoch,
                                         noise_levels=train_noise_levels)
        permutation = np.random.default_rng(seed + 901 + epoch).permutation(len(tr_y))
        model.train()
        losses = []
        for pos in range(0, len(permutation), batch_size):
            indices = permutation[pos:pos + batch_size]
            clean = torch.from_numpy(clean_train[indices]).to(device)
            noisy = torch.from_numpy(noisy_train[indices]).to(device)
            labels = torch.from_numpy(tr_y[indices]).to(device)
            optim.zero_grad(set_to_none=True)
            loss = paired_cross_entropy(model, clean, noisy, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite CE: {mode}/{seed}/epoch{epoch}")
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
        f1_clean = metrics(va_y, v2.inference(model, clean_val, device))["macro_f1"]
        f1_noisy = {str(level): metrics(va_y, v2.inference(model, x, device))["macro_f1"]
                    for level, x in fixed_noisy_val.items()}
        row = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
               "clean_f1": float(f1_clean), "noisy_f1": f1_noisy,
               "noise_mean_f1": float(np.mean(list(f1_noisy.values())))}
        # 记录整条记录的SNR分配数量，所有方法在同种子/轮次使用同一噪声。
        if train_protocol != "V2_M1_paired_source_CE":
            schedule = v2.training_noise_schedule(tr_raw, seed, epoch, train_noise_levels)
            row["train_snr_record_counts"] = {str(snr): sum(v == snr for v in schedule.values())
                                              for snr in train_noise_levels}
        history.append(row)
        states[epoch] = {key: value.detach().cpu().clone()
                         for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{mode}/seed{seed} epoch {epoch}/{epochs}: "
                  f"SrcVal Clean={f1_clean:.4f} Noisy={row['noise_mean_f1']:.4f}", flush=True)

    chosen = score_checkpoints(history)
    dest.mkdir(parents=True, exist_ok=False)
    ckpt = {"state": states[chosen["epoch"]], "mode": mode, "seed": seed,
            "selected": chosen, "source_only": True, "raw_points": RAW_POINTS,
            "code_family": VERSION, "initial_gain4": INITIAL_GAIN4,
            "train_protocol": train_protocol,
            "train_noise_levels_db": list(train_noise_levels),
            "val_noise_levels_db": list(val_noise_levels)}
    torch.save(ckpt, dest / "best.pt")
    (dest / "history.json").write_text(json.dumps(history, indent=2))
    (dest / "run.json").write_text(json.dumps({
        "mode": mode, "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "initial_gain4": INITIAL_GAIN4,
        "initial_cnn_sha256": initial_cnn_sha,
        "checkpoint_rule": selection_rule, "selected": chosen,
        "train_protocol": train_protocol,
        "train_noise_levels_db": list(train_noise_levels),
        "val_noise_levels_db": list(val_noise_levels),
        "source_split": "frozen_Gate0_V1_V2", "noise_policy": "V2_M1_full_record_per_phase_RMS",
        "parameter_counts": counts_by_module(model),
        "target_used_for_training_or_selection": False,
        "final_holdout_evaluated": False}, indent=2))
    print(f"SAVED {dest/'best.pt'}", flush=True)


def evaluate(mode: str, seed: int, output: Path, snr_root: Path | None,
             export_root: Path | None, conditions: list[int | None]) -> None:
    if mode not in METHODS:
        raise ValueError(mode)
    fix_randomness(seed)
    checkpoint_path = output / "models" / mode / f"seed{seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("code_family") != VERSION or checkpoint.get("source_only") is not True
            or checkpoint.get("mode") != mode or checkpoint.get("seed") != seed
            or checkpoint.get("raw_points") != RAW_POINTS):
        raise ValueError("Checkpoint version, split, mode, or seed mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPASFCNN(mode).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    destination = output / "evaluations" / mode / f"seed{seed}"
    for cond in conditions:
        name = "clean" if cond is None else f"snr{cond}"
        filepath = destination / f"target_dev_{name}.json"
        if filepath.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {filepath}")
        if export_root is not None:
            raw, meta = v2.target_dev_from_export(cond, export_root)
            source = "float32_dev_export"
        else:
            raw, meta = v2.target_dev_from_mat(cond, snr_root)
            source = "original_MAT"
        if len(meta) != 2474 or len({row["record_id"] for row in meta}) != 88:
            raise ValueError("Frozen 88-family/2474-cycle target Dev mismatch")
        y = np.asarray([row["label"] for row in meta], dtype=np.int64)
        x = v2.raw_cycles(raw, meta)
        prediction = v2.inference(model, x, device)
        result = {"method": mode, "seed": seed, "condition": name,
                  "record_families": 88, "windows": len(meta), "metrics": metrics(y, prediction),
                  "predicted": prediction.tolist(), "truth": y.tolist(),
                  "source_only_selection": checkpoint["selected"],
                  "data_source": source, "final_holdout_evaluated": False}
        destination.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(result, indent=2))
        print(f"DEV {mode}/seed{seed} {name} F1={100*result['metrics']['macro_f1']:.2f}%", flush=True)


def summarize(output: Path, methods: list[str], seeds: list[int],
              conditions: list[int | None]) -> dict:
    rows = []
    cache = {}
    for mode in methods:
        for seed in seeds:
            for condition in conditions:
                label = "clean" if condition is None else f"snr{condition}"
                f = output / "evaluations" / mode / f"seed{seed}" / f"target_dev_{label}.json"
                value = json.loads(f.read_text())
                if (value["condition"] != label or value["method"] != mode
                        or value["seed"] != seed or value["final_holdout_evaluated"] is not False
                        or value["record_families"] != 88 or value["windows"] != 2474):
                    raise ValueError(f"Invalid evaluation provenance: {f}")
                score = value["metrics"]
                row = {"method": mode, "seed": seed, "condition": label,
                       "macro_f1": 100*score["macro_f1"],
                       "accuracy": 100*score["accuracy"],
                       "normal_false_alarm_rate": 100*score["normal_false_alarm_rate"],
                       "missed_fault_as_normal": score["missed_fault_as_normal"]}
                rows.append(row)
                cache[mode, seed, label] = row
    aggregates = []
    pairs = []
    for mode in methods:
        for condition in conditions:
            label = "clean" if condition is None else f"snr{condition}"
            values = np.asarray([cache[mode, seed, label]["macro_f1"] for seed in seeds])
            aggregates.append({"method": mode, "condition": label,
                               "n_seeds": len(seeds),
                               "macro_f1_mean_percent": float(values.mean()),
                               "macro_f1_sd_percent": float(values.std(ddof=1)) if len(seeds)>1 else 0.})
    if "lowpass3" in methods:
        for mode in methods:
            if mode == "lowpass3":
                continue
            for condition in conditions:
                label = "clean" if condition is None else f"snr{condition}"
                delta = [cache[mode, seed, label]["macro_f1"] -
                         cache["lowpass3", seed, label]["macro_f1"] for seed in seeds]
                pairs.append({"method_minus_lowpass3": mode, "condition": label,
                              "paired_deltas_pp": delta, "mean_delta_pp": float(np.mean(delta)),
                              "positive_seeds": int(sum(v>0 for v in delta))})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "PER_SEED_RESULTS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"status": "V4_target_dev_only", "code_family": VERSION,
               "final_holdout_evaluated": False,
               "methods": methods, "seeds": seeds,
               "conditions": ["clean" if c is None else f"snr{c}" for c in conditions],
               "aggregates": aggregates, "paired_vs_lowpass3": pairs}
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"SAVED {output/'SUMMARY.json'}", flush=True)
    return summary


def smoke() -> None:
    fix_randomness(42)
    t = np.linspace(0, 1, 1001, dtype=np.float64)
    currents = np.stack([np.sin(2*np.pi*10*t + phase) for phase in np.arange(5)*.2], axis=1)
    raw = {"synthetic": (t, currents, np.full_like(t, 2*np.pi*10.))}
    metadata = [{"record_id": "synthetic", "label": 1, "start": 0.2, "end": 0.3}]
    x = v2.raw_cycles(raw, metadata)
    y = v2.fresh_noisy_source(raw, metadata, 42, 1)
    for mode in METHODS:
        model = HPASFCNN(mode)
        loss = paired_cross_entropy(model, torch.from_numpy(x), torch.from_numpy(y),
                                    torch.tensor([1]))
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(mode)
        if mode in ("static4", "hpasf"):
            grad = model.shaping.gate_logit.grad
            if grad is None or not torch.isfinite(grad) or grad.abs().sum() == 0:
                raise RuntimeError(f"Spectral parameter gradient missing: {mode}")
    print("V4 HPASF RAW-CYCLE TRAINING SMOKE PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "evaluate", "summarize"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--snr-root", type=Path)
    group.add_argument("--dev-export-root", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["clean","10","15","20","25","30","35","40"])
    args = parser.parse_args()
    conditions = [None if text == "clean" else int(text) for text in args.conditions]
    if any(value is not None and value not in NOISE_LEVELS for value in conditions):
        parser.error("Valid conditions: clean,10,15,20,25,30,35,40")
    if args.stage == "smoke":
        smoke()
    elif args.stage == "train":
        for mode in args.methods:
            for seed in args.seeds:
                train(mode, seed, args.output, args.epochs, args.batch, args.lr)
    elif args.stage == "evaluate":
        if args.dev_export_root is None and args.snr_root is None and any(c is not None for c in conditions):
            parser.error("Evaluation with noisy Dev requires --snr-root or --dev-export-root")
        for mode in args.methods:
            for seed in args.seeds:
                evaluate(mode, seed, args.output, args.snr_root, args.dev_export_root, conditions)
    elif args.stage == "summarize":
        summarize(args.output, args.methods, args.seeds, conditions)


if __name__ == "__main__":
    main()
