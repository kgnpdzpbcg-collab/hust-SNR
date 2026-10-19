"""Train/evaluate Jung Modified CNN and Hang WCNN on the frozen HUST-SNR protocol.

Run from repository root. This script reads only source Train/Val during training
and only the 88 frozen target Dev record families during evaluation. It never
loads target Final-Holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from jung_hang_models import (
    HangWCNN,
    JungModifiedCNN,
    current_window_to_vsd_images,
    five_phase_vsd,
)

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "outputs" / "gate0_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "jung_hang_comparison"
METHODS = ("jung", "hang")
SNR_LEVELS = (10, 15, 20, 25, 30, 35, 40)
VAL_SNR_LEVELS = (10, 20, 30)
SEEDS = (42, 43, 44)
NUM_CLASSES = 6


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_npz(name: str):
    path = PREPARED / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. First run the repository's approved preparation pipeline."
        )
    with np.load(path) as data:
        return data["x"].astype(np.float32), data["y"].astype(np.int64)


def load_source_raw(meta: list[dict]) -> dict:
    from prepare import record
    return {rid: record(rid, "source") for rid in sorted({row["record_id"] for row in meta})}


def add_white_noise_snr(currents: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Calibrate independent per-phase Gaussian noise to a requested raw-current SNR."""
    signal = np.asarray(currents, dtype=np.float64)
    noise = rng.standard_normal(signal.shape)
    signal_power = np.mean(signal ** 2, axis=0)
    noise_power = np.mean(noise ** 2, axis=0)
    scale = np.sqrt(np.maximum(signal_power, 1e-12) / np.maximum(noise_power, 1e-12))
    noise *= (scale / (10.0 ** (float(snr_db) / 20.0)))[None, :]
    return (signal + noise).astype(np.float64)


def noisy_raw_records(
    raw: Mapping[str, tuple],
    snr_by_record: Mapping[str, float],
    seed_token: str,
) -> dict:
    """Add noise once per full record, then reuse it for every overlapping window."""
    result = {}
    for rid in sorted(raw):
        time, currents, omega = raw[rid]
        rng = np.random.default_rng(stable_seed(f"{seed_token}:{rid}"))
        result[rid] = (time, add_white_noise_snr(currents, snr_by_record[rid], rng), omega)
    return result


def build_inputs(method: str, meta: list[dict], raw: Mapping[str, tuple]) -> np.ndarray:
    """Create [N,5,128] Jung inputs or [N,2,64,64] Hang trajectory images."""
    from prepare import preprocess
    values = []
    for row in meta:
        rid = row["record_id"]
        if rid not in raw:
            raise KeyError(f"Raw record {rid} required by frozen window metadata is missing")
        time, currents, _ = raw[rid]
        if method == "jung":
            value, _ = preprocess(time, currents, row["start"], row["end"])
        elif method == "hang":
            value = current_window_to_vsd_images(
                time, currents, row["start"], row["end"], points=128, image_size=64
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        values.append(value)
    result = np.stack(values).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Non-finite {method} inputs")
    return result


def make_train_noise_view(raw: Mapping[str, tuple], meta: list[dict], seed: int, epoch: int) -> np.ndarray:
    """Jung augmentation: random SNR per source record/epoch, before phase normalisation."""
    rng = np.random.default_rng(seed + 15000 + epoch)
    snr_by_record = {rid: float(rng.choice(SNR_LEVELS)) for rid in sorted(raw)}
    noised = noisy_raw_records(raw, snr_by_record, f"jung-train:{seed}:{epoch}")
    return build_inputs("jung", meta, noised)


def fixed_source_val_noise(
    method: str, meta: list[dict], raw: Mapping[str, tuple], snr: int
) -> np.ndarray:
    """Deterministic source-validation noise, identical across methods and training seeds."""
    snr_by_record = {rid: float(snr) for rid in raw}
    noised = noisy_raw_records(raw, snr_by_record, f"jh-source-val:{snr}")
    return build_inputs(method, meta, noised)


def get_model(method: str) -> nn.Module:
    if method == "jung":
        return JungModifiedCNN(in_channels=5, num_classes=NUM_CLASSES)
    if method == "hang":
        return HangWCNN(in_channels=2, num_classes=NUM_CLASSES, image_size=64)
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int = 128):
    model.eval()
    outputs = []
    for begin in range(0, len(values), batch_size):
        x = torch.from_numpy(values[begin:begin + batch_size]).to(device)
        outputs.append(model(x).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int64)


def classification_metrics(truth, prediction) -> dict:
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("truth/prediction must be equal-length one-dimensional arrays")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (truth, prediction), 1)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(
        2 * np.diag(cm), denom, out=np.zeros(NUM_CLASSES, dtype=np.float64), where=denom > 0
    )
    recall = np.divide(
        np.diag(cm), support, out=np.zeros(NUM_CLASSES, dtype=np.float64), where=support > 0
    )
    normal_count = int(support[0])
    normal_false_alarms = int(cm[0].sum() - cm[0, 0])
    missed_fault_as_normal = int(cm[1:, 0].sum())
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(np.trace(cm) / max(1, len(truth))),
        "balanced_accuracy": float(recall.mean()),
        "per_class_f1": f1.tolist(),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": cm.tolist(),
        "normal_false_alarms": normal_false_alarms,
        "normal_false_alarm_rate": float(normal_false_alarms / max(1, normal_count)),
        "missed_fault_as_normal": missed_fault_as_normal,
    }


def select_checkpoint(history: list[dict], clean_tolerance: float = 0.01) -> dict:
    """Shared source-only rule: clean within 1 pp of best, then best mean noisy-val F1."""
    if not history:
        raise ValueError("Empty validation history")
    max_clean = max(row["clean_val_macro_f1"] for row in history)
    eligible = [
        row for row in history
        if row["clean_val_macro_f1"] >= max_clean - clean_tolerance - 1e-12
    ]
    return max(
        eligible,
        key=lambda row: (
            row["noisy_val_mean_macro_f1"],
            row["clean_val_macro_f1"],
            -row["epoch"],
        ),
    )


def train_one(
    method: str,
    seed: int,
    output_dir: Path = DEFAULT_OUTPUT,
    epochs: int = 50,
    batch_size: int = 32,
):
    if method not in METHODS or epochs < 1 or batch_size < 1:
        raise ValueError("Invalid method, epochs, or batch_size")
    from prepare import read as prepare_read

    destination = output_dir / "models" / method / f"seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Refuse to overwrite existing run: {destination}")
    destination.mkdir(parents=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_saved, y_train = load_npz("source_train")
    x_val_saved, y_val = load_npz("source_val")
    train_meta = prepare_read(PREPARED / "source_train_windows.json")
    val_meta = prepare_read(PREPARED / "source_val_windows.json")
    if len(train_meta) != len(y_train) or len(val_meta) != len(y_val):
        raise RuntimeError("Prepared windows and labels do not align")

    raw_train = load_source_raw(train_meta)
    raw_val = load_source_raw(val_meta)
    if method == "jung":
        x_train = x_train_saved
        x_val = x_val_saved
    else:
        x_train = build_inputs(method, train_meta, raw_train)
        x_val = build_inputs(method, val_meta, raw_val)

    val_noise = {
        snr: fixed_source_val_noise(method, val_meta, raw_val, snr)
        for snr in VAL_SNR_LEVELS
    }
    model = get_model(method).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    history = []
    states = {}

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = np.random.default_rng(seed + 901 + epoch).permutation(len(y_train))
        aug = make_train_noise_view(raw_train, train_meta, seed, epoch) if method == "jung" else None
        losses = []
        for begin in range(0, len(permutation), batch_size):
            ids = permutation[begin:begin + batch_size]
            xb = torch.from_numpy(x_train[ids]).to(device)
            yb = torch.from_numpy(y_train[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            if method == "jung":
                x_noisy = torch.from_numpy(aug[ids]).to(device)
                logits = model(torch.cat((xb, x_noisy), dim=0))
                n = len(ids)
                loss = 0.5 * criterion(logits[:n], yb) + 0.5 * criterion(logits[n:], yb)
            else:
                loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {method}, seed={seed}, epoch={epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        clean_pred = predict(model, x_val, device)
        clean_f1 = classification_metrics(y_val, clean_pred)["macro_f1"]
        noisy_scores = {
            str(snr): classification_metrics(y_val, predict(model, x_noisy, device))["macro_f1"]
            for snr, x_noisy in val_noise.items()
        }
        row = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "clean_val_macro_f1": clean_f1,
            "noisy_val_macro_f1": noisy_scores,
            "noisy_val_mean_macro_f1": float(np.mean(list(noisy_scores.values()))),
        }
        history.append(row)
        states[epoch] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"{method} seed={seed} epoch={epoch}/{epochs} "
                f"loss={row['train_loss_mean']:.4f} cleanF1={clean_f1:.4f} "
                f"noisyF1={row['noisy_val_mean_macro_f1']:.4f}",
                flush=True,
            )

    best = select_checkpoint(history)
    torch.save(
        {
            "state": states[best["epoch"]],
            "method": method,
            "seed": seed,
            "epoch": best["epoch"],
            "selection": best,
            "selection_rule": (
                "source-only: clean validation Macro-F1 within 1 percentage point "
                "of the best clean epoch, then highest mean noisy validation Macro-F1 "
                "at 10/20/30 dB; ties by clean F1 then earlier epoch"
            ),
            "target_used_for_training_or_selection": False,
        },
        destination / "best.pt",
    )
    write_json(destination / "history.json", history)
    write_json(
        destination / "run.json",
        {
            "method": method,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "parameter_count": parameter_count,
            "train_windows": len(y_train),
            "source_val_windows": len(y_val),
            "source_train_records": len(raw_train),
            "source_val_records": len(raw_val),
            "best_epoch": best["epoch"],
            "best_source_clean_macro_f1": best["clean_val_macro_f1"],
            "best_source_noisy_macro_f1": best["noisy_val_mean_macro_f1"],
            "source_validation_snr_db": list(VAL_SNR_LEVELS),
            "target_dev_read": False,
            "target_final_holdout_read": False,
            "notes": (
                "Jung uses SNR-calibrated AWGN as a task-specific adaptation of the paper's "
                "THD+N augmentation; Hang uses a two-plane five-phase VSD trajectory-image "
                "adaptation. Neither is author-released code."
            ),
        },
    )
    print(f"Saved {destination / 'best.pt'}; selected epoch {best['epoch']}", flush=True)


def load_target_condition(rid: str, snr: int | None, snr_root: str | None):
    """Load a Clean or paired noisy Dev record; never resolves Final-Holdout paths."""
    from prepare import ROOT as PREPARE_ROOT, matrix, record

    clean_time, clean_current, clean_omega = record(rid, "target", "clean")
    if snr is None:
        return clean_time, clean_current, clean_omega
    base = Path(snr_root) if snr_root else PREPARE_ROOT / "data" / "target_dev"
    candidates = (
        base / f"snr{snr}" / f"{rid}.mat",
        base / f"SNR_{snr}" / f"{rid}.mat",
        base / str(snr) / f"{rid}.mat",
    )
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            f"Missing paired target Dev SNR-{snr} file for {rid}; searched {candidates}. "
            "Pass --snr-root pointing to the parent of SNR_10/.../SNR_40."
        )
    arr = matrix(found, "I_noisy")
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"Expected [time, five currents, omega] in {found}, got {arr.shape}")
    time, starts, counts = np.unique(arr[:, 0], return_index=True, return_counts=True)
    merged = np.add.reduceat(arr[:, 1:], starts, axis=0) / counts[:, None]
    if not np.array_equal(time, clean_time) or not np.allclose(
        merged[:, -1], clean_omega, rtol=0, atol=1e-12
    ):
        raise ValueError(f"Target SNR pairing/time-speed mismatch for {rid}")
    return time, merged[:, :5], merged[:, -1]


def evaluate_one(
    method: str,
    seed: int,
    output_dir: Path = DEFAULT_OUTPUT,
    snr_root: str | None = None,
):
    from prepare import read as prepare_read

    checkpoint = output_dir / "models" / method / f"seed{seed}" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    meta = prepare_read(PREPARED / "target_dev_clean_windows.json")
    record_ids = sorted({row["record_id"] for row in meta})
    if len(record_ids) != 88:
        raise RuntimeError(f"Expected the frozen 88 target Dev record families, got {len(record_ids)}")
    if set(record_ids) & set(prepare_read(ROOT / "configs" / "splits_gate0_v1.json")["target_holdout_records"]):
        raise RuntimeError("Target Dev metadata overlaps Final-Holdout")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(method).to(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state"], strict=True)
    truth = np.asarray([row["label"] for row in meta], dtype=np.int64)
    eval_dir = output_dir / "evaluations" / method / f"seed{seed}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[int | None] = [None, *SNR_LEVELS]
    for snr in conditions:
        name = "clean" if snr is None else f"snr{snr}"
        destination = eval_dir / f"target_dev_{name}.json"
        if destination.exists():
            raise FileExistsError(f"Refuse to overwrite evaluation: {destination}")
        raw = {rid: load_target_condition(rid, snr, snr_root) for rid in record_ids}
        values = build_inputs(method, meta, raw)
        prediction = predict(model, values, device)
        scores = classification_metrics(truth, prediction)
        write_json(
            destination,
            {
                "method": method,
                "seed": seed,
                "condition": name,
                "record_families": len(record_ids),
                "windows": len(truth),
                "metrics": scores,
                "truth": truth.tolist(),
                "predicted": prediction.tolist(),
                "selected_source_epoch": saved["epoch"],
                "selection_rule": saved["selection_rule"],
                "final_holdout_evaluated": False,
            },
        )
        print(
            f"DEV {method} seed={seed} {name}: "
            f"MacroF1={scores['macro_f1'] * 100:.2f}% "
            f"Acc={scores['accuracy'] * 100:.2f}% "
            f"normalFAR={scores['normal_false_alarm_rate'] * 100:.2f}%",
            flush=True,
        )


def summarize(
    output_dir: Path = DEFAULT_OUTPUT,
    methods: tuple[str, ...] = METHODS,
    seeds: tuple[int, ...] = SEEDS,
):
    conditions = ["clean", *[f"snr{snr}" for snr in SNR_LEVELS]]
    rows = []
    cache = {}
    for method in methods:
        for seed in seeds:
            for condition in conditions:
                path = output_dir / "evaluations" / method / f"seed{seed}" / f"target_dev_{condition}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing target Dev evaluation: {path}")
                data = read_json(path)
                if data.get("final_holdout_evaluated") is not False:
                    raise ValueError(f"Invalid evaluation provenance in {path}")
                scores = data["metrics"]
                row = {
                    "method": method,
                    "seed": seed,
                    "condition": condition,
                    "macro_f1_percent": 100 * float(scores["macro_f1"]),
                    "accuracy_percent": 100 * float(scores["accuracy"]),
                    "balanced_accuracy_percent": 100 * float(scores["balanced_accuracy"]),
                    "normal_false_alarm_rate_percent": 100 * float(scores["normal_false_alarm_rate"]),
                }
                rows.append(row)
                cache[method, seed, condition] = row
    aggregate = []
    metric_names = (
        "macro_f1_percent",
        "accuracy_percent",
        "balanced_accuracy_percent",
        "normal_false_alarm_rate_percent",
    )
    for method in methods:
        for condition in conditions:
            sample = [cache[method, seed, condition] for seed in seeds]
            row = {"method": method, "condition": condition, "n_seeds": len(seeds)}
            for key in metric_names:
                values = np.asarray([item[key] for item in sample], dtype=np.float64)
                row[key + "_mean"] = float(values.mean())
                row[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            aggregate.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "PER_SEED_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "SUMMARY.json",
        {
            "status": "target_dev_only",
            "methods": list(methods),
            "seeds": list(seeds),
            "conditions": conditions,
            "target_final_holdout_evaluated": False,
            "aggregates": aggregate,
        },
    )
    print(f"Saved {output_dir / 'SUMMARY.json'} and PER_SEED_RESULTS.csv", flush=True)


def smoke():
    """Synthetic-only architecture/gradient/transform smoke test; no dataset is read."""
    torch.manual_seed(42)
    x = torch.randn(4, 5, 128)
    jung = JungModifiedCNN()
    y = jung(x)
    assert y.shape == (4, 6) and torch.isfinite(y).all()
    nn.CrossEntropyLoss()(y, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in jung.parameters())

    image = torch.randn(4, 2, 64, 64)
    hang = HangWCNN()
    out = hang(image)
    assert out.shape == (4, 6) and torch.isfinite(out).all()
    nn.CrossEntropyLoss()(out, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in hang.parameters())
    assert all(not p.requires_grad for p in hang.pool1.parameters())

    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    theta = np.deg2rad(np.arange(5) * 72.0)
    current = np.stack([np.cos(t - angle) for angle in theta], axis=1)
    vsd = five_phase_vsd(current)
    assert vsd.shape == (4, 128) and np.isfinite(vsd).all()
    assert np.max(np.abs(vsd[2:])) < 1e-10
    trajectory = current_window_to_vsd_images(t, current, t[0], t[-1], points=128, image_size=64)
    assert trajectory.shape == (2, 64, 64) and np.isfinite(trajectory).all()
    print(
        "SMOKE PASSED",
        json.dumps(
            {
                "jung_parameters": sum(p.numel() for p in jung.parameters()),
                "hang_parameters": sum(p.numel() for p in hang.parameters()),
                "jung_logits": list(y.shape),
                "hang_logits": list(out.shape),
                "trajectory_shape": list(trajectory.shape),
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("smoke", help="Synthetic architecture and gradient checks only")

    train_parser = sub.add_parser("train", help="Train selected methods using source Train/Val only")
    train_parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    train_parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    eval_parser = sub.add_parser("evaluate", help="Evaluate frozen checkpoints on target Dev only")
    eval_parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    eval_parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    eval_parser.add_argument("--snr-root", type=str, default=None)
    eval_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    summary_parser = sub.add_parser("summarize", help="Aggregate paired-seed target Dev metrics")
    summary_parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    summary_parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    summary_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "train":
        for method in args.methods:
            for seed in args.seeds:
                train_one(method, seed, args.output, args.epochs, args.batch_size)
    elif args.command == "evaluate":
        for method in args.methods:
            for seed in args.seeds:
                evaluate_one(method, seed, args.output, args.snr_root)
    elif args.command == "summarize":
        summarize(args.output, tuple(args.methods), tuple(args.seeds))


if __name__ == "__main__":
    main()
