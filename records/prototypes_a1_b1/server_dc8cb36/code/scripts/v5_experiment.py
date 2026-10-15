"""V5 NA-HSF source-only training and 88-record Target-Dev evaluation.

V5 modifies ONLY the filter relative to V4. V4 training noise RNG, source
split, paired CE, Adam, 50 epochs, and clean-constrained source-val checkpoint
selection are reused without modification. 0/5 dB is *evaluation only*.

  python scripts/v5_experiment.py smoke
  python scripts/v5_experiment.py train
  python scripts/v5_experiment.py evaluate --snr-root /original/SNR_parent
  python scripts/v5_experiment.py evaluate --strong-snr-root /generated/target --conditions 5 0
  python scripts/v5_experiment.py summarize --conditions clean 10 15 20 25 30 35 40 5 0

Summarize compares existing V4 LP3/LP4 controls per seed and same frozen
Target-Dev windows.  No V4 checkpoints are touched or retrained.
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
from v5_nahsf import NAHSFCNN, parameter_counts

METHOD = "nahsf"
VERSION = "HUST-V5-NA-HSF-v0"
SEEDS = (42, 43, 44)
ORIGINAL_CONDITIONS = (None, 10, 15, 20, 25, 30, 35, 40)
SUPPORTED_CONDITIONS = (None, 0, 5, 10, 15, 20, 25, 30, 35, 40)
DEFAULT_OUTPUT = Path("outputs/v5_nahsf")
V4_ORIGINAL = Path("records/v4_hpasf/server_b4801df/results/evaluations")
V4_STRESS = Path("records/v4_strong_snr/server_84f6757/results/evaluations")
SELECT_RULE = "source 10/20/30 noisy-val mean Macro-F1; clean within 1pp of peak"


def _condition_name(value: int | None) -> str:
    return "clean" if value is None else f"snr{value}"


def _validate_train(epochs: int, batch: int, lr: float) -> None:
    if epochs < 1 or batch < 1 or not (0 < lr < 1):
        raise ValueError("epochs>=1, batch>=1, 0<lr<1 required")


def _initial_cnn_digest(model: NAHSFCNN) -> str:
    return hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters()
    )).hexdigest()


def train(seed: int, output: Path = DEFAULT_OUTPUT, epochs: int = 50,
          batch: int = 32, lr: float = .001) -> None:
    _validate_train(epochs, batch, lr)
    dest = output / "models" / METHOD / f"seed{seed}"
    if dest.exists():
        raise FileExistsError(f"Refuse overwrite: {dest}")
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_raw, tr_meta, y = v2.source_data("train")
    va_raw, va_meta, vy = v2.source_data("val")
    clean_train = v2.raw_cycles(tr_raw, tr_meta)
    clean_val = v2.raw_cycles(va_raw, va_meta)
    fixed_val = v2.fixed_noisy_validation(va_raw, va_meta)  # 10/20/30 only
    if list(sorted(fixed_val)) != list(sorted(VAL_NOISE_LEVELS)):
        raise RuntimeError("Frozen V4 source validation noise schedule changed")
    model = NAHSFCNN().to(device)
    initial_cnn_sha = _initial_cnn_digest(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    states: dict[int, dict] = {}
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        noisy_train = v2.fresh_noisy_source(tr_raw, tr_meta, seed, epoch)
        permutation = np.random.default_rng(seed + 901 + epoch).permutation(len(y))
        model.train()
        losses = []
        gate_grad = []
        for pos in range(0, len(permutation), batch):
            ids = permutation[pos:pos + batch]
            cx = torch.from_numpy(clean_train[ids]).to(device)
            nx = torch.from_numpy(noisy_train[ids]).to(device)
            labels = torch.from_numpy(y[ids]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = paired_cross_entropy(model, cx, nx, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite training loss: seed{seed} epoch{epoch}")
            loss.backward()
            gate_last = model.shaping.gate[-1]
            grad = gate_last.bias.grad
            if grad is None or not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("NA-HSF gate gradient missing / nonfinite")
            gate_grad.append(float(grad.detach().abs().mean().cpu()))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        f1_clean = metrics(vy, v2.inference(model, clean_val, device))["macro_f1"]
        f1_noisy = {str(level): metrics(vy, v2.inference(model, x, device))["macro_f1"]
                    for level, x in fixed_val.items()}
        history.append({
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "gate_bias_grad_abs_mean": float(np.mean(gate_grad)),
            "gate_bias_value": float(model.shaping.gate[-1].bias.detach().cpu().item()),
            "clean_f1": float(f1_clean),
            "noisy_f1": f1_noisy,
            "noise_mean_f1": float(np.mean(list(f1_noisy.values()))),
        })
        states[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{METHOD}/seed{seed}: epoch {epoch}/{epochs}, "
                  f"src Clean={f1_clean:.4f}, noisy={history[-1]['noise_mean_f1']:.4f}", flush=True)
    chosen = score_checkpoints(history)
    dest.mkdir(parents=True, exist_ok=False)
    selected_state = states[chosen["epoch"]]
    checkpoint = {
        "state": selected_state, "method": METHOD, "seed": seed,
        "selected": chosen, "source_only": True, "raw_points": RAW_POINTS,
        "code_family": VERSION, "train_protocol": "V4_original_noise_10_to_40",
        "train_noise_levels_db": list(NOISE_LEVELS),
        "val_noise_levels_db": list(VAL_NOISE_LEVELS),
        "gate_features": 3, "gate_hidden": 8,
        "final_holdout_evaluated": False,
    }
    torch.save(checkpoint, dest / "best.pt")
    (dest / "history.json").write_text(json.dumps(history, indent=2))
    (dest / "run.json").write_text(json.dumps({
        "method": METHOD, "seed": seed, "epochs": epochs, "batch": batch,
        "learning_rate": lr, "checkpoint_rule": SELECT_RULE,
        "selected": chosen, "initial_cnn_sha256": initial_cnn_sha,
        "parameter_counts": parameter_counts(model),
        "gate_bias_grad_abs_mean_over_epochs": float(np.mean([r["gate_bias_grad_abs_mean"] for r in history])),
        "train_noise_levels_db": list(NOISE_LEVELS),
        "val_noise_levels_db": list(VAL_NOISE_LEVELS),
        "target_used_for_training_or_checkpoint_selection": False,
        "final_holdout_evaluated": False,
    }, indent=2))
    print(f"SAVED {dest/'best.pt'}", flush=True)


@torch.no_grad()
def predict_with_gate(model: NAHSFCNN, x: np.ndarray, device: torch.device,
                      batch_size: int = 128):
    model.eval()
    all_pred, all_gate, all_descriptor = [], [], []
    for start in range(0, len(x), batch_size):
        raw = torch.from_numpy(x[start:start + batch_size]).to(device)
        scores, alpha, descriptor = model(raw, return_descriptors=True)
        all_pred.append(scores.argmax(dim=-1).cpu().numpy())
        all_gate.append(alpha.cpu().numpy())
        all_descriptor.append(descriptor.cpu().numpy())
    return (np.concatenate(all_pred), np.concatenate(all_gate),
            np.concatenate(all_descriptor))


def gate_summary(alpha: np.ndarray, descriptor: np.ndarray,
                 truth: np.ndarray) -> dict:
    if alpha.shape != truth.shape or descriptor.shape != (len(truth), 3):
        raise ValueError("Gate descriptor / truth length mismatch")
    if not np.all(np.isfinite(alpha)) or not np.all((alpha >= 0) & (alpha <= 1)):
        raise ValueError("Gate alpha invalid")
    percentiles = np.quantile(alpha, [.1, .5, .9])
    return {
        "mean": float(alpha.mean()), "std": float(alpha.std()),
        "min": float(alpha.min()), "max": float(alpha.max()),
        "p10": float(percentiles[0]), "p50": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "by_true_class_mean": [float(alpha[truth == k].mean())
                               if np.any(truth == k) else None for k in range(6)],
        "descriptor_mean": descriptor.mean(axis=0).astype(float).tolist(),
    }


def evaluate(seed: int, output: Path, snr_root: Path | None,
             strong_snr_root: Path | None, conditions: list[int | None]) -> None:
    if any(c not in SUPPORTED_CONDITIONS for c in conditions):
        raise ValueError("Only Clean or SNR=0,5,10,15,...,40 supported")
    # 与原V4推理设置一致：启用严格确定性，关闭TF32；每个独立评估进程
    # 都显式设置，不能依赖前一次训练进程留下的状态。模型和输入不改变。
    fix_randomness(seed)
    ckpt_path = output / "models" / METHOD / f"seed{seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if (ckpt.get("code_family") != VERSION or ckpt.get("method") != METHOD or
        ckpt.get("seed") != seed or ckpt.get("source_only") is not True or
        ckpt.get("raw_points") != RAW_POINTS or
        ckpt.get("train_protocol") != "V4_original_noise_10_to_40" or
        ckpt.get("val_noise_levels_db") != list(VAL_NOISE_LEVELS)):
        raise ValueError("Wrong V5 source-only checkpoint / protocol")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NAHSFCNN().to(device)
    model.load_state_dict(ckpt["state"], strict=True)
    dest = output / "evaluations" / METHOD / f"seed{seed}"
    for cond in conditions:
        name = _condition_name(cond)
        path = dest / f"target_dev_{name}.json"
        if path.exists():
            raise FileExistsError(f"Refuse overwrite: {path}")
        if cond in (0, 5):
            if strong_snr_root is None:
                raise ValueError("Synthetic 0/5 dB requires --strong-snr-root")
            root = strong_snr_root
            source = "synthetic_scaled_original_snr10"
        elif cond is None:
            root = None
            source = "original_clean_MAT"
        else:
            if snr_root is None:
                raise ValueError("Original SNR data requires --snr-root")
            root = snr_root
            source = "original_published_snr_MAT"
        raw, meta = v2.target_dev_from_mat(cond, root)
        if len(meta) != 2474 or len({r["record_id"] for r in meta}) != 88:
            raise ValueError("Frozen target Dev window or family mismatch")
        truth = np.asarray([r["label"] for r in meta], dtype=np.int64)
        cycle = v2.raw_cycles(raw, meta)
        prediction, alpha, descriptors = predict_with_gate(model, cycle, device)
        report = {
            "method": METHOD, "seed": seed, "condition": name,
            "record_families": 88, "windows": len(meta),
            "metrics": metrics(truth, prediction),
            "predicted": prediction.tolist(), "truth": truth.tolist(),
            "gate": gate_summary(alpha, descriptors, truth),
            "source_only_selection": ckpt["selected"],
            "data_source": source,
            "target_used_for_training_or_checkpoint_selection": False,
            "final_holdout_evaluated": False,
        }
        dest.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"V5 DEV seed{seed} {name} Macro-F1={100*report['metrics']['macro_f1']:.2f}% "
              f"alpha_mean={report['gate']['mean']:.3f}", flush=True)


def _read_eval(path: Path, mode: str, seed: int, condition: str) -> dict:
    data = json.loads(path.read_text())
    # 已发布的V4独立压力测试以snr_db保存0/5 dB，原V4以condition保存。
    # 只解释这一明确历史格式，不改写旧JSON；若两个字段同时存在且冲突，
    # 仍应拒绝，避免错误配对其他噪声条件的成绩。
    actual_condition = data.get("condition")
    legacy_snr = data.get("snr_db")
    if mode in ("lowpass3", "lowpass4") and legacy_snr in (0, 5):
        legacy_condition = _condition_name(legacy_snr)
        if actual_condition is None:
            actual_condition = legacy_condition
        elif actual_condition != legacy_condition:
            raise ValueError(f"Evaluation condition/snr_db conflict: {path}")
    if (actual_condition != condition or data.get("seed") != seed or
        data.get("method") != mode or data.get("record_families") != 88 or
        data.get("windows") != 2474 or
        data.get("final_holdout_evaluated") is not False or
        len(data.get("truth", [])) != 2474 or
        len(data.get("predicted", [])) != 2474):
        raise ValueError(f"Evaluation provenance mismatch: {path}")
    return data


def summarize(output: Path, seeds: list[int], conditions: list[int | None],
              v4_original: Path = V4_ORIGINAL,
              v4_stress: Path = V4_STRESS) -> dict:
    if not seeds or not conditions:
        raise ValueError("Must specify seeds and conditions")
    aggregates, per_seed = [], []
    for cond in conditions:
        name = _condition_name(cond)
        rows = []
        for seed in seeds:
            new_file = output / "evaluations" / METHOD / f"seed{seed}" / f"target_dev_{name}.json"
            new = _read_eval(new_file, METHOD, seed, name)
            for old_method in ("lowpass3", "lowpass4"):
                old_root = v4_stress if cond in (0, 5) else v4_original
                old_file = old_root / old_method / f"seed{seed}" / f"target_dev_{name}.json"
                old = _read_eval(old_file, old_method, seed, name)
                if new["truth"] != old["truth"]:
                    raise ValueError(f"Wrong target window order or labels: {old_file}")
                truth = np.asarray(new["truth"], dtype=np.int64)
                p, q = np.asarray(new["predicted"]), np.asarray(old["predicted"])
                pair = {
                    "condition": name, "seed": seed, "vs": old_method,
                    "v5_macro_f1": 100*new["metrics"]["macro_f1"],
                    "v4_macro_f1": 100*old["metrics"]["macro_f1"],
                    "delta_pp": 100*(new["metrics"]["macro_f1"]-old["metrics"]["macro_f1"]),
                    "v5_only_correct": int(((p == truth) & (q != truth)).sum()),
                    "v4_only_correct": int(((p != truth) & (q == truth)).sum()),
                    "gate_mean": new["gate"]["mean"],
                    "gate_p10": new["gate"]["p10"],
                    "gate_p90": new["gate"]["p90"],
                }
                rows.append(pair)
                per_seed.append(pair)
        for old_method in ("lowpass3", "lowpass4"):
            paired = [r for r in rows if r["vs"] == old_method]
            delta = np.asarray([r["delta_pp"] for r in paired], dtype=float)
            aggregates.append({
                "condition": name, "vs": old_method,
                "seeds": seeds, "mean_delta_pp": float(delta.mean()),
                "sd_delta_pp": float(delta.std(ddof=1)) if len(delta)>1 else 0.,
                "positive_seeds": int((delta > 0).sum()),
                "v5_mean_macro_f1": float(np.mean([r["v5_macro_f1"] for r in paired])),
                "v4_mean_macro_f1": float(np.mean([r["v4_macro_f1"] for r in paired])),
                "v5_alpha_mean": float(np.mean([r["gate_mean"] for r in paired])),
            })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "PAIRED_V4_COMPARISON.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    summary = {"status": "V5_dev_only_comparison", "code_family": VERSION,
               "frozen_v4_baselines": {"original": str(v4_original), "stress": str(v4_stress)},
               "seeds": seeds, "conditions": [_condition_name(c) for c in conditions],
               "aggregates": aggregates, "final_holdout_evaluated": False}
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"SAVED {output/'SUMMARY.json'}")
    return summary


def smoke():
    fix_randomness(42)
    t = np.linspace(0, 1, 1001, dtype=np.float64)
    currents = np.stack([np.sin(2*np.pi*10*t + p) for p in np.arange(5)*.2], axis=1)
    raw = {"synthetic": (t, currents, np.full_like(t, 2*np.pi*10))}
    metadata = [{"record_id": "synthetic", "label": 1, "start": .2, "end": .3}]
    clean = v2.raw_cycles(raw, metadata)
    noisy = v2.fresh_noisy_source(raw, metadata, seed=42, epoch=1)
    model = NAHSFCNN()
    loss = paired_cross_entropy(model, torch.from_numpy(clean),
                                torch.from_numpy(noisy), torch.tensor([1]))
    loss.backward()
    assert bool(torch.isfinite(loss))
    print("V5 NA-HSF SOURCE PIPELINE SMOKE PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("smoke", "train", "evaluate", "summarize"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=.001)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--snr-root", type=Path, help="Parent of published SNR_10,...,SNR_40")
    ap.add_argument("--strong-snr-root", type=Path, help="Parent of generated SNR_0 and SNR_5")
    ap.add_argument("--conditions", nargs="+", default=[
        "clean", "10", "15", "20", "25", "30", "35", "40"])
    ap.add_argument("--v4-original-results", type=Path, default=V4_ORIGINAL)
    ap.add_argument("--v4-stress-results", type=Path, default=V4_STRESS)
    args = ap.parse_args()
    conditions = [None if text.lower() == "clean" else int(text) for text in args.conditions]
    if len(set(conditions)) != len(conditions) or any(c not in SUPPORTED_CONDITIONS for c in conditions):
        ap.error("Conditions must be unique and from clean,0,5,10,15,20,25,30,35,40")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in SEEDS for seed in args.seeds):
        ap.error("Use distinct frozen seeds 42,43,44")
    if args.stage == "smoke":
        smoke()
    elif args.stage == "train":
        for seed in args.seeds:
            train(seed, args.output, args.epochs, args.batch, args.lr)
    elif args.stage == "evaluate":
        for seed in args.seeds:
            evaluate(seed, args.output, args.snr_root, args.strong_snr_root, conditions)
    else:
        summarize(args.output, args.seeds, conditions,
                  args.v4_original_results, args.v4_stress_results)


if __name__ == "__main__":
    main()
