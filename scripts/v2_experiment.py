"""Source-only HUST V2 training and 88-record Dev evaluation for PaiFilter ablation.

Run in the hust-SNR repository.  No target-domain training, adaptation, or
checkpoint selection. Never load target Final-Holdout from this entrypoint.

python scripts/v2_paifilter.py
python scripts/v2_experiment.py smoke
python scripts/v2_experiment.py train
python scripts/v2_experiment.py evaluate --snr-root /path/to/SNR_parent
python scripts/v2_experiment.py summarize

A data adapter for the user's 88-record dev export is also supported:
python scripts/v2_experiment.py evaluate --dev-export-root /path/to/dev_export
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

from v2_paifilter import (HarmonicCNN, METHODS, RAW_POINTS, counts_by_module,
                          paired_cross_entropy)
from v1_noise_consistency import (NOISE_LEVELS, VAL_NOISE_LEVELS, fix_randomness,
                                   metrics, score_checkpoints, target_condition_record)

DEFAULT_OUT = Path("outputs/v2_paifilter")


def raw_cycles(raw_records: dict, metadata: list[dict], points: int = RAW_POINTS) -> np.ndarray:
    """Raw unnormalized 5-phase currents; linear interpolation to cycle grid.

    Important: labels are read only from existing frozen metadata, never from
    the signal to determine segmentation boundaries.
    """
    result = np.empty((len(metadata), 5, points), dtype=np.float32)
    grid = np.linspace(0., 1., points, dtype=np.float64)
    for pos, row in enumerate(metadata):
        rid = row["record_id"]
        t, current, _ = raw_records[rid]
        start, end = float(row["start"]), float(row["end"])
        if not 0 <= start < end <= t[-1] + 1e-8:
            raise ValueError(f"Window outside raw record: {rid}")
        sample_times = start + grid * (end - start)
        for channel in range(5):
            result[pos, channel] = np.interp(sample_times, t, current[:, channel])
    if not np.isfinite(result).all():
        raise ValueError("Non-finite cycle samples")
    return result


def calibrated_noise(raw_records: dict, snr_by_record: dict,
                     rng_by_record) -> dict:
    """V1 M1's full-record, per-phase RMS calibrated Gaussian noise policy.

    This generates NOISE before cycle selection and MD normalization, just as
    V1 did. A shared input realization is created for all V2 model variants
    when seed+epoch match.
    """
    modified = {}
    for rid, (times, current, omega) in raw_records.items():
        rng = rng_by_record(rid)
        eps = rng.standard_normal(current.shape)
        power = np.mean(current.astype(np.float64) ** 2, axis=0)
        eps *= np.sqrt(power / np.mean(eps ** 2, axis=0)) / (10 ** (snr_by_record[rid] / 20))
        modified[rid] = (times, current + eps, omega)
    return modified


def fresh_noisy_source(raw_records: dict, metadata: list[dict], seed: int,
                       epoch: int, points: int = RAW_POINTS) -> np.ndarray:
    """Matches V1's seed/epoch noise generator including per-record ordering."""
    rng = np.random.default_rng(seed + 15000 + epoch)
    ids = sorted(raw_records)
    snr = {rid: int(rng.choice(NOISE_LEVELS)) for rid in ids}
    generators = {rid: np.random.default_rng(seed + 15000 + epoch + 997 * (i + 1))
                  for i, rid in enumerate(ids)}
    noisy = calibrated_noise(raw_records, snr, lambda rid: generators[rid])
    return raw_cycles(noisy, metadata, points)


def fixed_noisy_validation(raw_records: dict, metadata: list[dict],
                           points: int = RAW_POINTS) -> dict[int, np.ndarray]:
    """Held-out SOURCE simulation Val, independent fixed 10/20/30 dB draws."""
    val = {}
    for snr in VAL_NOISE_LEVELS:
        by_rec = {rid: snr for rid in raw_records}
        def rng(rid):
            digest = hashlib.sha256(f"HUST-V1:val:{snr}:{rid}".encode()).digest()
            return np.random.default_rng(int.from_bytes(digest[:8], "little"))
        val[snr] = raw_cycles(calibrated_noise(raw_records, by_rec, rng), metadata, points)
    return val


def source_data(part: str):
    from prepare import OUT, read, record
    if part not in ("train", "val"):
        raise ValueError(part)
    metadata = read(OUT / f"source_{part}_windows.json")
    if len(metadata) != (738 if part == "train" else 302):
        raise ValueError("Frozen V1 source split mismatch")
    with np.load(OUT / f"source_{part}.npz") as f:
        target_labels = f["y"].astype(np.int64)
    y = np.array([row["label"] for row in metadata], dtype=np.int64)
    if not np.array_equal(target_labels, y):
        raise ValueError("Frozen source labels mismatch")
    raw = {rid: record(rid, "source") for rid in sorted({row["record_id"] for row in metadata})}
    return raw, metadata, y


@torch.no_grad()
def inference(model, raw: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    prediction = []
    for i in range(0, len(raw), batch_size):
        scores = model(torch.from_numpy(raw[i:i + batch_size]).to(device))
        prediction.append(scores.argmax(dim=1).cpu().numpy())
    return np.concatenate(prediction)


def training_run(mode: str, seed: int, outdir: Path, epochs: int = 50,
                 batch_size: int = 32, lr: float = 0.001) -> None:
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid training hyperparameters")
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = outdir / "models" / mode / f"seed{seed}"
    if folder.exists():
        raise FileExistsError(f"Refuse to overwrite prior run: {folder}")
    train_raw, train_meta, y = source_data("train")
    val_raw, val_meta, vy = source_data("val")
    cx = raw_cycles(train_raw, train_meta)
    vx = raw_cycles(val_raw, val_meta)
    fixed = fixed_noisy_validation(val_raw, val_meta)
    model = HarmonicCNN(mode).to(device)
    initial_sha = hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    history, states = [], {}
    for epoch in range(1, epochs + 1):
        noisy = fresh_noisy_source(train_raw, train_meta, seed, epoch)
        perm = np.random.default_rng(seed + 901 + epoch).permutation(len(y))
        model.train()
        losses = []
        for i in range(0, len(perm), batch_size):
            ids = perm[i:i + batch_size]
            xb = torch.from_numpy(cx[ids]).to(device)
            nb = torch.from_numpy(noisy[ids]).to(device)
            label = torch.from_numpy(y[ids]).to(device)
            optim.zero_grad(set_to_none=True)
            loss = paired_cross_entropy(model, xb, nb, label)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
        clean_f1 = metrics(vy, inference(model, vx, device))["macro_f1"]
        noisy_f1 = {str(snr): metrics(vy, inference(model, values, device))["macro_f1"]
                    for snr, values in fixed.items()}
        item = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
                "clean_f1": clean_f1, "noisy_f1": noisy_f1,
                "noise_mean_f1": float(np.mean(list(noisy_f1.values())))}
        history.append(item)
        states[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{mode} seed={seed} epoch={epoch}/{epochs} "
                  f"source_Clean_F1={clean_f1:.4f} noisy_F1={item['noise_mean_f1']:.4f}", flush=True)
    best = score_checkpoints(history)
    folder.mkdir(parents=True, exist_ok=False)
    torch.save({"state": states[best["epoch"]], "mode": mode, "seed": seed,
                "selected": best, "source_only": True, "raw_points": RAW_POINTS,
                "train_protocol": "paired-source-CE"}, folder / "best.pt")
    (folder / "history.json").write_text(json.dumps(history, indent=2))
    (folder / "run.json").write_text(json.dumps({
        "mode": mode, "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": lr, "first_epoch_clean_noise_identical_across_modes": True,
        "selection": best, "checkpoint_rule": "source 10/20/30 dB noisy-val mean Macro-F1, clean within 1pp",
        "source_split": "frozen Gate0/V1", "initial_cnn_sha256": initial_sha,
        "train_or_select_on_target": False, "final_holdout_evaluated": False,
        "parameter_counts": counts_by_module(model)
    }, indent=2))
    print(f"SAVED {folder/'best.pt'}", flush=True)


def target_dev_from_mat(snr: int | None, snr_root: Path | None):
    from prepare import OUT, read
    meta = read(OUT / "target_dev_clean_windows.json")
    ids = sorted({row["record_id"] for row in meta})
    if len(ids) != 88:
        raise ValueError("Expected exactly 88 frozen Dev raw record families")
    return {rid: target_condition_record(rid, snr, snr_root) for rid in ids}, meta


def target_dev_from_export(snr: int | None, root: Path):
    """Read *only* provided 88-record user export, never source training."""
    name = "clean" if snr is None else f"snr{snr}"
    with np.load(root / "dev_reference.npz", allow_pickle=False) as ref:
        ids = ref["record_ids"].astype(str).tolist()
        t = ref["time"].copy()
        omega = ref["omega"].copy()
    with np.load(root / f"dev_{name}.npz", allow_pickle=False) as db:
        currents = db["currents"].copy()
        indices = db["record_indices"].copy()
    if len(ids) != 88 or set(indices.tolist()) != set(range(88)) or currents.shape != (88, 10011, 5):
        raise ValueError("Dev export integrity or completeness failure")
    raw = {rid: (t[i], currents[np.flatnonzero(indices == i)[0]], omega[i])
           for i, rid in enumerate(ids)}
    meta = json.loads((root / "dev_cycle_windows.json").read_text())
    if len(meta) != 2474 or {x["record_id"] for x in meta} != set(ids):
        raise ValueError("Exported frozen Dev window mismatch")
    return raw, meta


def evaluation_run(mode: str, seed: int, outdir: Path, snr_root: Path | None,
                   export_root: Path | None, conditions: list[int | None]):
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = outdir / "models" / mode / f"seed{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["mode"] != mode or checkpoint["seed"] != seed or not checkpoint["source_only"]:
        raise ValueError("Checkpoint metadata mismatch")
    model = HarmonicCNN(mode).to(device)
    model.load_state_dict(checkpoint["state"], strict=True)
    results_dir = outdir / "evaluations" / mode / f"seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    for snr in conditions:
        label = "clean" if snr is None else f"snr{snr}"
        if export_root is not None:
            raw, meta = target_dev_from_export(snr, export_root)
            source = "float32_dev_export"
        else:
            raw, meta = target_dev_from_mat(snr, snr_root)
            source = "original_MAT"
        window_labels = np.array([x["label"] for x in meta], dtype=np.int64)
        x = raw_cycles(raw, meta)
        pred = inference(model, x, device)
        result = {"method": mode, "seed": seed, "condition": label,
                  "record_families": 88, "windows": len(meta), "metrics": metrics(window_labels, pred),
                  "predicted": pred.tolist(), "truth": window_labels.tolist(),
                  "data_source": source, "source_only_selection": checkpoint["selected"],
                  "final_holdout_evaluated": False}
        (results_dir / f"target_dev_{label}.json").write_text(json.dumps(result, indent=2))
        print(f"DEV {mode} seed={seed} {label} Macro-F1={100*result['metrics']['macro_f1']:.2f}%", flush=True)


def summarize(outdir: Path, modes: list[str], seeds: list[int], conditions: list[int | None]):
    rows = []
    for mode in modes:
        for seed in seeds:
            for snr in conditions:
                cond = "clean" if snr is None else f"snr{snr}"
                path = outdir / "evaluations" / mode / f"seed{seed}" / f"target_dev_{cond}.json"
                item = json.loads(path.read_text())
                if item["final_holdout_evaluated"] is not False or item["condition"] != cond:
                    raise ValueError(f"Invalid result: {path}")
                rows.append({"mode": mode, "seed": seed, "condition": cond,
                             "macro_f1": 100*item["metrics"]["macro_f1"],
                             "accuracy": 100*item["metrics"]["accuracy"],
                             "normal_false_alarm_rate": 100*item["metrics"]["normal_false_alarm_rate"],
                             "missed_fault_as_normal": item["metrics"]["missed_fault_as_normal"]})
    aggregated = []
    for mode in modes:
        for snr in conditions:
            cond = "clean" if snr is None else f"snr{snr}"
            items = [row for row in rows if row["mode"] == mode and row["condition"] == cond]
            scores = np.array([r["macro_f1"] for r in items])
            aggregated.append({"mode": mode, "condition": cond, "n_seeds": len(scores),
                               "macro_f1_mean_percent": float(scores.mean()),
                               "macro_f1_sd_percent": float(scores.std(ddof=1)) if len(scores)>1 else 0.0})
    comparisons = []
    if "paifilter" in modes and "identity" in modes:
        for snr in conditions:
            cond = "clean" if snr is None else f"snr{snr}"
            diffs = [next(r["macro_f1"] for r in rows if r["mode"]=="paifilter" and r["seed"]==seed and r["condition"]==cond)-
                     next(r["macro_f1"] for r in rows if r["mode"]=="identity" and r["seed"]==seed and r["condition"]==cond)
                     for seed in seeds]
            comparisons.append({"condition": cond, "paifilter_minus_identity_pp": diffs,
                                "mean_pp": float(np.mean(diffs))})
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "PER_SEED_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (outdir / "SUMMARY.json").write_text(json.dumps({"status": "V2_dev_only",
        "final_holdout_evaluated": False, "aggregates": aggregated,
        "paired_paifilter_minus_identity": comparisons}, indent=2))
    print(f"SAVED {outdir/'SUMMARY.json'}")


def smoke():
    fix_randomness(42)
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 1001)
    raw = {"test": (t, np.stack([np.sin((i+1)*2*np.pi*10*t) for i in range(5)], axis=1),
                    np.full_like(t, 20.))}
    meta = [{"record_id": "test", "label": 1, "start": 0.2, "end": 0.3}]
    x = raw_cycles(raw, meta)
    y = fresh_noisy_source(raw, meta, 42, 1)
    for mode in METHODS:
        model = HarmonicCNN(mode)
        loss = paired_cross_entropy(model, torch.from_numpy(x), torch.from_numpy(y),
                                    torch.tensor([1]))
        loss.backward()
        assert torch.isfinite(loss)
        if mode == "paifilter":
            g = model.shaping.raw_attenuation.grad
            assert g is not None and g.abs().sum() > 0
    print("V2 TRAIN/RAW-CYCLE SMOKE PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "evaluate", "summarize"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--snr-root", type=Path)
    parser.add_argument("--dev-export-root", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["clean","10","15","20","25","30","35","40"])
    args = parser.parse_args()
    cond = [None if value == "clean" else int(value) for value in args.conditions]
    if any(value is not None and value not in NOISE_LEVELS for value in cond):
        raise ValueError("Choose clean,10,15,20,25,30,35,40")
    if args.stage == "smoke":
        smoke()
    elif args.stage == "train":
        for mode in args.methods:
            for seed in args.seeds:
                training_run(mode, seed, args.output, args.epochs, args.batch, args.lr)
    elif args.stage == "evaluate":
        for mode in args.methods:
            for seed in args.seeds:
                evaluation_run(mode, seed, args.output, args.snr_root, args.dev_export_root, cond)
    else:
        summarize(args.output, args.methods, args.seeds, cond)


if __name__ == "__main__":
    main()
