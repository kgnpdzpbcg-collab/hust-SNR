"""Independent HUST A1/B1 source-only experiment runner.

Commands (from repo root):
  python scripts/hust_prototypes_experiment.py smoke
  python scripts/hust_prototypes_experiment.py train --methods a1 b1 --source-strong-root ROOT
  python scripts/hust_prototypes_experiment.py evaluate --methods a1 b1 \
      --strong-root /path/old_generated/target --extreme-root /path/new_extreme_root
  python scripts/hust_prototypes_experiment.py summarize --methods a1 b1

The runner NEVER reads target data during training or checkpoint selection.
No source test/final holdout loader is defined; only the frozen 88-family Dev.
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

import v2_experiment as data
from hust_arch_suite_experiment import (
    source_views, infer, digest_parameters, write_json, write_csv,
)
from v1_noise_consistency import fix_randomness, metrics
from torch.nn import functional as F
from hust_a1_reliability_wavelet import A1NoiseAwareWaveletCNN, a1_paired_loss
from hust_b1_phase_harmonic import (
    B1PhaseAwareRecoveryNet, b1_paired_loss, b1_recovery_objective,
)
from hust_b0_harmonic_recovery import count_parameters

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hust_prototypes_a1_b1_v1.json"
OUT = Path("outputs/hust_prototypes_a1_b1_v1")
METHODS = ("a1", "b1")
SEEDS = (42, 43, 44)
TRAIN_NOISE = (10, 15, 20, 25, 30, 35, 40)
VAL_NOISE = (0, 5, 10, 20, 30)
CONDITIONS = (5, 0, -5, -10)
PROTOCOL = "hust-a1-b1-source-only-v1"


def get_config() -> dict:
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "train_noise_db": list(TRAIN_NOISE),
        "val_noise_db": list(VAL_NOISE),
        "main_target_conditions_db": list(CONDITIONS),
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "b1_pretrain_epochs": 10,
        "b1_restore_weight": 1.0,
        "selection_rule": "source-clean within one pp peak; highest source five-noise mean Macro-F1; then higher clean F1; when tied, smallest mean five-noise source CE; then earliest epoch",
        "source_only": True,
        "final_holdout_evaluated": False,
    }
    for k, v in expected.items():
        if c.get(k) != v:
            raise ValueError(f"Frozen research config changed: {k}")
    return c


def build_model(method: str) -> torch.nn.Module:
    if method == "a1":
        return A1NoiseAwareWaveletCNN()
    if method == "b1":
        return B1PhaseAwareRecoveryNet()
    raise ValueError(f"Unsupported method {method}")


def _snapshot(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


@torch.no_grad()
def source_cross_entropy(model: torch.nn.Module, x: np.ndarray,
                         y: np.ndarray, device: torch.device,
                         batch_size: int = 128) -> float:
    """Source-only CE used only to disambiguate fully saturated source Macro-F1."""
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Invalid source validation batch")
    model.eval()
    summed = 0.0
    for start in range(0, len(x), batch_size):
        bx = torch.from_numpy(x[start:start + batch_size]).to(device)
        by = torch.from_numpy(y[start:start + batch_size]).to(device)
        logits = model(bx)
        summed += float(F.cross_entropy(logits, by, reduction="sum").cpu())
    return summed / len(y)


def select_source_checkpoint(history: list[dict], tolerance: float = 0.01) -> dict:
    """Keep original F1 rule, using SOURCE CE only when F1 cannot rank epochs.

    No target-domain input; source CE is not guaranteed to predict real-domain
    transfer performance (not a claim of solved model-selection generalization).
    """
    if not history or not (0 <= tolerance < 1):
        raise ValueError("Invalid selection history / tolerance")
    maximum = max(row["clean_f1"] for row in history)
    eligible = [row for row in history
                if row["clean_f1"] >= maximum - tolerance - 1e-12]
    return max(eligible, key=lambda row: (
        row["noise_mean_f1"], row["clean_f1"],
        -row["noise_mean_ce"], -row["epoch"]))


def source_pretrain_b1(model: B1PhaseAwareRecoveryNet, tr: dict, tm: list,
                       clean: np.ndarray, seed: int, config: dict,
                       device: torch.device) -> list[float]:
    optimizer = torch.optim.Adam(model.restorer.parameters(),
                                 lr=config["learning_rate"], weight_decay=0)
    history = []
    for epoch in range(1, config["b1_pretrain_epochs"] + 1):
        noisy = data.fresh_noisy_source(tr, tm, seed, epoch,
                                        noise_levels=TRAIN_NOISE)
        perm = np.random.default_rng(seed + 11001 + epoch).permutation(len(clean))
        model.restorer.train()
        losses = []
        for pos in range(0, len(perm), config["batch_size"]):
            idx = perm[pos:pos + config["batch_size"]]
            nx = torch.from_numpy(noisy[idx]).to(device)
            cx = torch.from_numpy(clean[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            _, spec, variance, _ = model.restorer(nx, return_details=True)
            loss, _ = b1_recovery_objective(nx, cx, spec, variance)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"B1 pretraining nonfinite epoch {epoch}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all().item()
                       for p in model.restorer.parameters()):
                raise FloatingPointError("B1 pretraining nonfinite gradients")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        if epoch in (1, config["b1_pretrain_epochs"]):
            print(f"B1 source recovery epoch {epoch}: {history[-1]:.6f}", flush=True)
    return history


def train(method: str, seed: int, out: Path, source_strong_root: Path) -> None:
    config = get_config()
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Method or seed not permitted")
    destination = out / "models" / method / f"seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    fix_randomness(seed)
    (tr, tm, y, clean), (_, _, vy, val_clean, val_noise) = source_views(source_strong_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(method).to(device)
    initial_sha = digest_parameters(model)
    pretrain_history = []
    if method == "b1":
        pretrain_history = source_pretrain_b1(
            model, tr, tm, clean, seed, config, device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["learning_rate"], weight_decay=0)
    history, states = [], {}
    for epoch in range(1, config["epochs"] + 1):
        noisy = data.fresh_noisy_source(tr, tm, seed, epoch,
                                        noise_levels=TRAIN_NOISE)
        if noisy.shape != clean.shape:
            raise ValueError("Frozen noisy training cycle mismatch")
        perm = np.random.default_rng(seed + 901 + epoch).permutation(len(y))
        model.train()
        losses, ce_parts, restore_parts = [], [], []
        for start in range(0, len(perm), config["batch_size"]):
            idx = perm[start:start + config["batch_size"]]
            clean_x = torch.from_numpy(clean[idx]).to(device)
            noisy_x = torch.from_numpy(noisy[idx]).to(device)
            labels = torch.from_numpy(y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            if method == "b1":
                loss, components = b1_paired_loss(
                    model, clean_x, noisy_x, labels,
                    lambda_restore=config["b1_restore_weight"])
                ce_parts.append(float(components["ce"].cpu()))
                restore_parts.append(float(components["restore"].cpu()))
            else:
                loss = a1_paired_loss(model, clean_x, noisy_x, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss {method}/{seed}/{epoch}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all().item()
                       for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        clean_f1 = metrics(vy, infer(model, val_clean, device))["macro_f1"]
        noisy_f1 = {
            str(snr): metrics(vy, infer(model, val_noise[snr], device))["macro_f1"]
            for snr in VAL_NOISE
        }
        noisy_ce = {
            str(snr): source_cross_entropy(model, val_noise[snr], vy, device)
            for snr in VAL_NOISE
        }
        entry = {
            "epoch": epoch, "clean_f1": float(clean_f1),
            "noisy_f1": noisy_f1,
            "noise_mean_f1": float(np.mean(list(noisy_f1.values()))),
            "noisy_ce": noisy_ce,
            "noise_mean_ce": float(np.mean(list(noisy_ce.values()))),
            "train_loss_mean": float(np.mean(losses)),
        }
        if method == "b1":
            entry["classification_ce_mean"] = float(np.mean(ce_parts))
            entry["source_restore_loss_mean"] = float(np.mean(restore_parts))
        history.append(entry)
        states[epoch] = _snapshot(model)
        if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
            print(f"{method} seed{seed} epoch {epoch}/{config['epochs']}: "
                  f"source-val F1={entry['noise_mean_f1']:.4f} "
                  f"CE={entry['noise_mean_ce']:.6f}", flush=True)
    chosen = select_source_checkpoint(history)
    destination.mkdir(parents=True, exist_ok=False)
    torch.save({
        "state": states[chosen["epoch"]],
        "method": method, "seed": seed, "protocol": PROTOCOL,
        "selected": chosen, "config": config, "source_only": True,
        "final_holdout_evaluated": False,
    }, destination / "best.pt")
    write_json(destination / "history.json", history)
    write_json(destination / "run.json", {
        "method": method, "seed": seed, "protocol": PROTOCOL,
        "initial_parameters_sha256": initial_sha,
        "parameters": count_parameters(model),
        "selected": chosen, "pretrain_recovery_losses": pretrain_history,
        "config": config, "source_train_cycles": 738,
        "source_val_cycles": 302,
        "target_used_for_training_or_selection": False,
        "final_holdout_evaluated": False,
    })



def _target_cycles(snr: int, *, ordinary_root: Path | None,
                   extreme_root: Path | None, export_root: Path | None):
    if export_root is not None:
        return data.target_dev_from_export(snr, export_root)
    root = ordinary_root if snr in (0, 5) else extreme_root
    if root is None:
        raise ValueError(f"No root for target SNR={snr} dB")
    # Both roots point DIRECTLY to folders containing SNR_<value> directories.
    return data.target_dev_from_mat(snr, root)


def evaluate(method: str, seed: int, out: Path, conditions: list[int], *,
             ordinary_root: Path | None, extreme_root: Path | None,
             export_root: Path | None) -> None:
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Invalid method/seed")
    from prepare import OUT as PREPARE_OUT, read
    frozen = read(PREPARE_OUT / "target_dev_clean_windows.json")
    if len(frozen) != 2474 or len({r["record_id"] for r in frozen}) != 88:
        raise ValueError("Frozen Target-Dev cohort changed")
    labels = np.array([r["label"] for r in frozen], dtype=np.int64)
    folder = out / "models" / method / f"seed{seed}"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    if (run["method"], run["seed"], run["protocol"]) != (method, seed, PROTOCOL):
        raise ValueError("Run provenance invalid")
    if (saved["method"], saved["seed"], saved["protocol"]) != (method, seed, PROTOCOL):
        raise ValueError("Checkpoint provenance invalid")
    if not saved["source_only"] or saved["final_holdout_evaluated"] or \
            run["target_used_for_training_or_selection"]:
        raise ValueError("Target data leakage marker")
    if saved["config"] != get_config():
        raise ValueError("Incompatible checkpoint configuration")
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(method).to(device)
    model.load_state_dict(saved["state"], strict=True)
    if count_parameters(model) != run["parameters"]:
        raise ValueError("Architecture parameter mismatch")
    for snr in conditions:
        if snr not in CONDITIONS:
            raise ValueError(f"Unsupported SNR: {snr}")
        target = out / "evaluations" / method / f"seed{seed}" / f"target_dev_snr{snr}.json"
        if target.exists():
            raise FileExistsError(f"Refuse overwrite: {target}")
        raws, meta = _target_cycles(snr, ordinary_root=ordinary_root,
                                   extreme_root=extreme_root, export_root=export_root)
        if meta != frozen:
            raise ValueError("Target record/window identities changed")
        x = data.raw_cycles(raws, meta)
        if x.shape != (2474, 5, 4096):
            raise ValueError("Target cycle tensor shape mismatch")
        pred = infer(model, x, device)
        report = {
            "method": method, "seed": seed, "protocol": PROTOCOL,
            "condition": f"snr{snr}", "windows": len(labels),
            "record_families": 88, "truth": labels.tolist(),
            "predicted": pred.tolist(),
            "metrics": metrics(labels, pred),
            "source_only_selection": saved["selected"],
            "noise_source": "10dB_author_residual_scaled_to_requested_SNR",
            "target_used_for_training_or_checkpoint_selection": False,
            "final_holdout_evaluated": False,
        }
        write_json(target, report)
        print(f"{method} seed{seed} {snr}dB: "
              f"Macro-F1={report['metrics']['macro_f1']*100:.2f}%", flush=True)


def summarize(out: Path, methods: list[str], seeds: list[int], conditions: list[int]) -> None:
    per_seed, per_class, summary = [], [], []
    common_truth = None
    for method in methods:
        for seed in seeds:
            for snr in conditions:
                name = f"snr{snr}"
                path = out / "evaluations" / method / f"seed{seed}" / f"target_dev_{name}.json"
                obj = json.loads(path.read_text(encoding="utf-8"))
                if (obj["method"], obj["seed"], obj["protocol"], obj["condition"]) != \
                        (method, seed, PROTOCOL, name):
                    raise ValueError("Evaluation file provenance mismatch")
                if obj["record_families"] != 88 or obj["windows"] != 2474:
                    raise ValueError("Target cohort count mismatch")
                if obj["final_holdout_evaluated"]:
                    raise ValueError("Final holdout was accessed")
                if common_truth is None:
                    common_truth = obj["truth"]
                if common_truth != obj["truth"]:
                    raise ValueError("Different target evaluation cohorts")
                m = obj["metrics"]
                per_seed.append({
                    "method": method, "seed": seed, "condition": name,
                    "macro_f1_percent": 100*m["macro_f1"],
                    "accuracy_percent": 100*m["accuracy"],
                    "fault_as_healthy": m["missed_fault_as_normal"],
                    "healthy_as_fault": m["normal_false_alarms"],
                })
                for cls, (f1, rec) in enumerate(zip(m["per_class_f1"], m["per_class_recall"])):
                    per_class.append({
                        "method": method, "seed": seed, "condition": name,
                        "class": cls, "f1_percent": 100*f1, "recall_percent": 100*rec,
                    })
    for method in methods:
        for snr in conditions:
            name = f"snr{snr}"
            rows = [x for x in per_seed if x["method"] == method and x["condition"] == name]
            if len(rows) != len(seeds):
                raise ValueError("Incomplete seed results")
            f1 = np.array([x["macro_f1_percent"] for x in rows])
            acc = np.array([x["accuracy_percent"] for x in rows])
            summary.append({
                "method": method, "condition": name, "seeds": len(rows),
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_sd": float(f1.std(ddof=1)) if len(rows)>1 else 0.0,
                "accuracy_mean": float(acc.mean()),
                "accuracy_sd": float(acc.std(ddof=1)) if len(rows)>1 else 0.0,
            })
    write_csv(out / "reports" / "PER_SEED_RESULTS.csv", per_seed)
    write_csv(out / "reports" / "PER_CLASS_PER_SEED.csv", per_class)
    write_csv(out / "reports" / "SUMMARY.csv", summary)
    write_json(out / "reports" / "SUMMARY.json", {
        "protocol": PROTOCOL, "methods": methods, "seeds": seeds,
        "conditions": conditions, "final_holdout_evaluated": False,
        "summary": summary,
    })


def smoke() -> None:
    torch.manual_seed(42)
    raw = torch.randn(2, 5, 4096)
    noisy = raw + 0.1 * torch.randn_like(raw)
    labels = torch.tensor([1, 4])
    for method in METHODS:
        model = build_model(method)
        if method == "b1":
            loss, _ = b1_paired_loss(model, raw, noisy, labels)
        else:
            loss = a1_paired_loss(model, raw, noisy, labels)
        loss.backward()
        if not torch.isfinite(loss).item() or not any(
                p.grad is not None and torch.isfinite(p.grad).all().item()
                for p in model.parameters()):
            raise AssertionError(f"Failed gradient check: {method}")
        print(f"SMOKE {method}: parameters={count_parameters(model)}, "
              f"loss={loss.item():.6f}")



def main():
    get_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "train", "evaluate", "summarize"))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--source-strong-root", type=Path)
    p.add_argument("--strong-root", type=Path, help="Existing 5/0dB parent with SNR_5, SNR_0")
    p.add_argument("--extreme-root", type=Path, help="New -5/-10dB parent with SNR_-5, SNR_-10")
    p.add_argument("--dev-export-root", type=Path)
    p.add_argument("--conditions", nargs="+", type=int, default=list(CONDITIONS))
    args = p.parse_args()
    if not args.methods or len(args.methods) != len(set(args.methods)):
        p.error("Duplicate/empty methods")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or any(
            seed not in SEEDS for seed in args.seeds):
        p.error("Seeds must be a unique subset of 42/43/44")
    if not args.conditions or len(set(args.conditions)) != len(args.conditions) or \
            any(c not in CONDITIONS for c in args.conditions):
        p.error("Conditions must be a unique subset of 5,0,-5,-10")
    if args.stage == "smoke":
        smoke()
    elif args.stage == "train":
        if args.source_strong_root is None:
            p.error("train requires --source-strong-root")
        for method in args.methods:
            for seed in args.seeds:
                train(method, seed, args.out, args.source_strong_root)
    elif args.stage == "evaluate":
        if args.dev_export_root is None:
            if any(c in (5, 0) for c in args.conditions) and args.strong_root is None:
                p.error("5/0dB requires --strong-root")
            if any(c in (-5, -10) for c in args.conditions) and args.extreme_root is None:
                p.error("-5/-10dB requires --extreme-root")
        for method in args.methods:
            for seed in args.seeds:
                evaluate(method, seed, args.out, args.conditions,
                         ordinary_root=args.strong_root,
                         extreme_root=args.extreme_root,
                         export_root=args.dev_export_root)
    else:
        summarize(args.out, args.methods, args.seeds, args.conditions)


if __name__ == "__main__":
    main()
