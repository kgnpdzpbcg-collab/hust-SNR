"""V1: paired source-only CNN / Gaussian augmentation / noise-consistency experiment.

Train on simulated labeled waveforms only; select checkpoints on simulated
clean+fixed-noise validation; evaluate the same 88 target Dev record families
on Clean and seven SNR versions. Never read the target Final-Holdout.

Dependencies: existing scripts/prepare.py and gate0_v1 preparation artifacts.
Run from the repository root; see docs/V1_NOISE_CONSISTENCY_PROTOCOL.md.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("baseline", "augmentation", "consistency")
NOISE_LEVELS = (10, 15, 20, 25, 30, 35, 40)
VAL_NOISE_LEVELS = (10, 20, 30)
CLASSES = 6


class CNN(nn.Module):
    """Exactly the gate0 CNN topology, with optional 50-d penultimate features."""
    def __init__(self):
        super().__init__()
        layers = []
        for inc, out, kernel in ((5, 16, 16), (16, 32, 4), (32, 32, 4)):
            layers.extend([nn.Conv1d(inc, out, kernel), nn.BatchNorm1d(out),
                           nn.LeakyReLU(.01), nn.MaxPool1d(2)])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(352, 50),
                                        nn.LeakyReLU(.01), nn.Linear(50, CLASSES))

    def forward(self, x, return_features=False):
        h = self.classifier[:3](self.features(x))
        logits = self.classifier[3](h)
        return (logits, h) if return_features else logits


def consistency_loss(z_clean, z_noisy):
    """Paired cosine distance; no stop-gradient, with CE preventing trivial collapse."""
    if z_clean.shape != z_noisy.shape:
        raise ValueError("Clean and noisy feature batch sizes differ")
    return (1.0 - F.cosine_similarity(z_clean, z_noisy, dim=1, eps=1e-8)).mean()


def train_objective(model, x, y, noisy=None, method="baseline", lam=0.1):
    if method not in METHODS:
        raise ValueError(method)
    if method == "baseline":
        if noisy is not None:
            raise ValueError("Baseline must not take a noisy view")
        return F.cross_entropy(model(x), y), {"ce": None, "cons": 0.0}
    if noisy is None or noisy.shape != x.shape:
        raise ValueError("Two-view methods need exactly paired inputs")
    logits, feats = model(torch.cat([x, noisy], dim=0), return_features=True)
    n = len(y)
    ce = (F.cross_entropy(logits[:n], y) + F.cross_entropy(logits[n:], y)) / 2
    cons = consistency_loss(feats[:n], feats[n:])
    objective = ce + (lam * cons if method == "consistency" else 0)
    return objective, {"ce": float(ce.detach()), "cons": float(cons.detach())}


def metrics(y, prediction):
    truth, pred = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    cm = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(cm, (truth, pred), 1)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    denom = support + predicted
    per_f1 = np.divide(2 * np.diag(cm), denom, out=np.zeros(CLASSES, float), where=denom > 0)
    per_rec = np.divide(np.diag(cm), support, out=np.zeros(CLASSES, float), where=support > 0)
    normal_fp = int(predicted[0] - cm[0, 0])
    normal_fn = int(support[0] - cm[0, 0])
    return {"macro_f1": float(per_f1.mean()), "accuracy": float(np.trace(cm) / max(1, len(truth))),
            "balanced_accuracy": float(per_rec.mean()), "per_class_f1": per_f1.tolist(),
            "per_class_recall": per_rec.tolist(), "normal_false_alarms": normal_fn,
            "normal_false_alarm_rate": float(normal_fn / max(1, support[0])),
            "missed_fault_as_normal": normal_fp, "confusion_matrix": cm.tolist()}


def fix_randomness(seed):
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


def load_npz(path):
    with np.load(path) as f:
        return f["x"].astype(np.float32), f["y"].astype(np.int64)


def load_source_raw(names):
    from prepare import record
    return {name: record(name, "source") for name in sorted(set(names))}


def noisy_view(raw, meta, snr_by_record, rng_by_record):
    """Add calibrated Gaussian noise to raw full records, THEN window/normalize."""
    from prepare import preprocess
    prepared = {}
    for rid, (time, current, _) in raw.items():
        rng = rng_by_record(rid)
        noise = rng.standard_normal(current.shape)
        power = np.mean(current ** 2, axis=0)
        noise *= (np.sqrt(power / np.mean(noise ** 2, axis=0)) / (10 ** (snr_by_record[rid] / 20)))
        prepared[rid] = current + noise
    result = []
    for row in meta:
        rid = row["record_id"]
        time = raw[rid][0]
        result.append(preprocess(time, prepared[rid], row["start"], row["end"])[0])
    return np.stack(result).astype(np.float32)


def fixed_source_noise_val(meta, levels=VAL_NOISE_LEVELS):
    """Fixed held-out synthetic noise per source Val record, independent of train seeds."""
    raw = load_source_raw(row["record_id"] for row in meta)
    output = {}
    for snr in levels:
        fixed = {rid: snr for rid in raw}
        def stable_rng(rid):
            digest = hashlib.sha256(f"HUST-V1:val:{snr}:{rid}".encode()).digest()
            return np.random.default_rng(int.from_bytes(digest[:8], "little"))
        output[snr] = noisy_view(raw, meta, fixed, stable_rng)
    return output


def fresh_train_noise(raw, meta, epoch, seed):
    """Same per-record random augmentation policy as gate0; same views for M1/M2."""
    rng = np.random.default_rng(seed + 15000 + epoch)
    by_record = {rid: int(rng.choice(NOISE_LEVELS)) for rid in sorted(raw)}
    generators = {rid: np.random.default_rng(seed + 15000 + epoch + 997 * (i + 1))
                  for i, rid in enumerate(sorted(raw))}
    # The raw noise realization is fixed per method/seed/epoch by independent RNGs.
    return noisy_view(raw, meta, by_record, lambda rid: generators[rid])


@torch.no_grad()
def predictions(model, x, device, batch=256):
    model.eval()
    out = []
    for pos in range(0, len(x), batch):
        data = torch.from_numpy(x[pos:pos + batch]).to(device)
        out.append(model(data).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def score_checkpoints(history, tolerance=0.01):
    """Choose max noisy-source-Val F1 among epochs within 1 pp best clean F1.

    Deterministic, applies identically to all three methods, uses no target labels.
    """
    max_clean = max(row["clean_f1"] for row in history)
    eligible = [row for row in history if row["clean_f1"] >= max_clean - tolerance - 1e-12]
    return max(eligible, key=lambda row: (row["noise_mean_f1"], row["clean_f1"], -row["epoch"]))


def train(method, seed, outdir, lam=0.1, epochs=50, batch_size=32):
    if method not in METHODS or lam < 0:
        raise ValueError("Invalid method/lambda")
    from prepare import OUT, read
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dst = Path(outdir) / "models" / method / f"seed{seed}"
    if dst.exists():
        raise FileExistsError(f"Refuse overwrite: {dst}")
    dst.mkdir(parents=True)
    x, y = load_npz(OUT / "source_train.npz")
    vx, vy = load_npz(OUT / "source_val.npz")
    tr_meta = read(OUT / "source_train_windows.json")
    va_meta = read(OUT / "source_val_windows.json")
    if len(tr_meta) != len(x) or len(va_meta) != len(vx):
        raise RuntimeError("Preprocessing metadata mismatch")
    fixed_noise = fixed_source_noise_val(va_meta)
    source_raw = load_source_raw(row["record_id"] for row in tr_meta) if method != "baseline" else {}
    model = CNN().to(device)
    initial_sha = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    stored_states, history = {}, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.default_rng(seed + 901 + epoch).permutation(len(x))
        aug = fresh_train_noise(source_raw, tr_meta, epoch, seed) if source_raw else None
        batch_loss = []
        for pos in range(0, len(perm), batch_size):
            ids = perm[pos:pos + batch_size]
            data = torch.from_numpy(x[ids]).to(device)
            labels = torch.from_numpy(y[ids]).to(device)
            noisy = torch.from_numpy(aug[ids]).to(device) if aug is not None else None
            optim.zero_grad(set_to_none=True)
            loss, _ = train_objective(model, data, labels, noisy, method, lam)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optim.step()
            batch_loss.append(float(loss.detach()))
        clean_f1 = metrics(vy, predictions(model, vx, device))["macro_f1"]
        noisy_f1 = {str(k): metrics(vy, predictions(model, v, device))["macro_f1"]
                    for k, v in fixed_noise.items()}
        row = {"epoch": epoch, "train_loss_mean": float(np.mean(batch_loss)),
               "clean_f1": clean_f1, "noisy_f1": noisy_f1,
               "noise_mean_f1": float(np.mean(list(noisy_f1.values())))}
        history.append(row)
        stored_states[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{method} seed{seed} {epoch}/{epochs}: src clean={clean_f1:.4f} noisy={row['noise_mean_f1']:.4f}", flush=True)
    best = score_checkpoints(history)
    state = stored_states[best["epoch"]]
    torch.save({"state": state, "method": method, "seed": seed, "epoch": best["epoch"],
                "lambda_consistency": lam, "selection": best,
                "selection_rule": "best source noisy-val mean F1 subject to clean-val within 1pp of best clean-val"}, dst / "best.pt")
    (dst / "history.json").write_text(json.dumps(history, indent=2))
    (dst / "run.json").write_text(json.dumps({"method": method, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "lambda_consistency": lam, "initial_param_sha256": initial_sha,
        "chosen": best, "target_used_in_training_or_selection": False,
        "selection_source_noise_db": list(VAL_NOISE_LEVELS)}, indent=2))
    print(f"SAVED {dst / 'best.pt'} (epoch {best['epoch']})", flush=True)


def target_condition_record(rid, snr, root=None):
    """Use Clean time/speed and evaluate exact same windows for all SNRs."""
    from prepare import ROOT, record, matrix
    clean_time, clean_i, clean_w = record(rid, "target", "clean")
    if snr is None:
        return clean_time, clean_i, clean_w
    base = Path(root) if root else ROOT / "data" / "target_dev"
    options = [base / f"snr{snr}" / f"{rid}.mat", base / f"SNR_{snr}" / f"{rid}.mat",
               base / f"{snr}" / f"{rid}.mat"]
    found = [p for p in options if p.is_file()]
    if not found:
        raise FileNotFoundError(f"SNR_{snr} file missing for {rid}; searched {options}. "
                                "Provide --snr-root (parent of SNR_10/.../SNR_40).")
    arr = matrix(found[0], "I_noisy")
    # Match existing prepare.record handling of repeated target timestamps.
    t, start, count = np.unique(arr[:, 0], return_index=True, return_counts=True)
    merged = np.add.reduceat(arr[:, 1:], start, axis=0) / count[:, None]
    if not np.array_equal(t, clean_time) or not np.allclose(merged[:, -1], clean_w, rtol=0, atol=1e-12):
        raise ValueError(f"Target SNR pair does not match Clean timing/speed: {rid}")
    return t, merged[:, :5], merged[:, -1]


def evaluate(method, seed, outdir, snr_root=None, conditions=None):
    from prepare import OUT, read, preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(outdir) / "models" / method / f"seed{seed}" / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model = CNN().to(device)
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state"], strict=True)
    meta = read(OUT / "target_dev_clean_windows.json")
    ids = sorted(set(row["record_id"] for row in meta))
    if len(ids) != 88:
        raise ValueError(f"Expected exact frozen 88 Dev record families; got {len(ids)}")
    evaldir = Path(outdir) / "evaluations" / method / f"seed{seed}"
    evaldir.mkdir(parents=True, exist_ok=True)
    conditions = conditions or [None, *NOISE_LEVELS]
    results = {}
    for snr in conditions:
        name = "clean" if snr is None else f"snr{snr}"
        rows = {rid: target_condition_record(rid, snr, snr_root) for rid in ids}
        values, labels = [], []
        for row in meta:
            t, currents, _ = rows[row["record_id"]]
            z, _ = preprocess(t, currents, row["start"], row["end"])
            values.append(z)
            labels.append(row["label"])
        pred = predictions(model, np.stack(values).astype(np.float32), device)
        m = metrics(np.asarray(labels), pred)
        results[name] = {"method": method, "seed": seed, "condition": name,
                         "record_families": len(ids), "windows": len(labels), "metrics": m,
                         "predicted": pred.tolist(), "truth": labels,
                         "source_only_selection": saved["selection"],
                         "final_holdout_evaluated": False}
        (evaldir / f"target_dev_{name}.json").write_text(json.dumps(results[name], indent=2))
        print(f"DEV {method} seed{seed} {name}: F1={m['macro_f1']*100:.2f}% acc={m['accuracy']*100:.2f}% normalFAR={m['normal_false_alarm_rate']*100:.2f}%", flush=True)
    return results


def smoke():
    fix_randomness(42)
    model = CNN()
    clean = torch.randn(6, 5, 128)
    noisy = clean + 0.2 * torch.randn_like(clean)
    label = torch.tensor([0, 1, 2, 3, 4, 5])
    losses = {}
    for m in METHODS:
        model.zero_grad(set_to_none=True)
        loss, aux = train_objective(model, clean, label, None if m == "baseline" else noisy, m, .1)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
        losses[m] = float(loss.detach())
    assert len(model(clean, return_features=True)[1][0]) == 50
    assert consistency_loss(model(clean, True)[1], model(clean, True)[1]).abs() < 1e-6
    print("SMOKE PASSED", json.dumps(losses))


def summarize(outdir, methods=METHODS, seeds=(42, 43, 44), conditions=None):
    """Aggregate frozen Dev scores and paired M2-M1 deltas; no model retraining."""
    conditions = conditions or ["clean", *[f"snr{s}" for s in NOISE_LEVELS]]
    root = Path(outdir)
    rows, aggregates, contrasts = [], [], []
    cache = {}
    for method in methods:
        for seed in seeds:
            for cond in conditions:
                source = root / "evaluations" / method / f"seed{seed}" / f"target_dev_{cond}.json"
                if not source.is_file():
                    raise FileNotFoundError(f"Missing official target Dev result: {source}")
                data = json.loads(source.read_text())
                if data.get("final_holdout_evaluated") is not False or data.get("condition") != cond:
                    raise ValueError(f"Invalid evaluation provenance: {source}")
                scores = data["metrics"]
                row = {"method": method, "seed": seed, "condition": cond,
                       **{k: 100 * float(scores[k]) for k in
                          ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate")}}
                rows.append(row)
                cache[method, seed, cond] = row
    for method in methods:
        for cond in conditions:
            sample = [cache[method, seed, cond] for seed in seeds]
            summary = {"method": method, "condition": cond, "n_seeds": len(seeds)}
            for k in ("macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"):
                z = np.array([row[k] for row in sample])
                summary[k+"_mean_percent"] = float(z.mean())
                summary[k+"_sd_percent"] = float(z.std(ddof=1)) if len(z)>1 else 0.0
            aggregates.append(summary)
    if "augmentation" in methods and "consistency" in methods:
        for cond in conditions:
            diffs = [cache["consistency", seed, cond]["macro_f1"]-
                     cache["augmentation", seed, cond]["macro_f1"] for seed in seeds]
            contrasts.append({"condition": cond, "m2_minus_m1_macro_f1_pp": diffs,
                              "mean_pp": float(np.mean(diffs)),
                              "sd_pp": float(np.std(diffs, ddof=1)) if len(diffs)>1 else 0.0,
                              "positive_seeds": int(sum(x>0 for x in diffs))})
    root.mkdir(parents=True, exist_ok=True)
    with (root / "PER_SEED_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    out = {"status": "V1_target_dev_only", "final_holdout_evaluated": False,
           "methods": list(methods), "seeds": list(seeds), "conditions": list(conditions),
           "aggregates": aggregates, "paired_consistency_minus_augmentation": contrasts}
    (root / "SUMMARY.json").write_text(json.dumps(out, indent=2))
    print(f"SAVED {root/'SUMMARY.json'} and PER_SEED_RESULTS.csv", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "evaluate", "summarize"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default="outputs/v1_noise_consistency")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lambda-cons", dest="lam", type=float, default=0.1)
    parser.add_argument("--snr-root", type=Path)
    parser.add_argument("--conditions", nargs="+", default=["clean", "10", "15", "20", "25", "30", "35", "40"])
    a = parser.parse_args()
    if a.stage == "smoke":
        return smoke()
    if a.stage == "train":
        for m in a.methods:
            for seed in a.seeds:
                train(m, seed, a.output, a.lam, a.epochs, a.batch)
    elif a.stage == "summarize":
        names = ["clean" if x == "clean" else f"snr{int(x)}" for x in a.conditions]
        summarize(a.output, a.methods, a.seeds, names)
    elif a.stage == "evaluate":
        c = [None if x == "clean" else int(x) for x in a.conditions]
        if any(x is not None and x not in NOISE_LEVELS for x in c):
            raise ValueError("Unsupported SNR: use clean,10,15,20,25,30,35,40")
        for m in a.methods:
            for seed in a.seeds:
                evaluate(m, seed, a.output, a.snr_root, c)


if __name__ == "__main__":
    main()
    main()