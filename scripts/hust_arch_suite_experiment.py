"""Frozen HUST nine-arm architecture sweep, source-only training/selection.

Reuses V5's source-data, noise schedules, labels, metric, and checkpoint selector.
No Source-Test or Final-Holdout access. All new methods are trained from
scratch. Historical A'/V5/ResNet results are NEVER overwritten or retrained.

Usage (repository root):
  python scripts/hust_arch_suite_experiment.py smoke
  python scripts/hust_arch_suite_experiment.py train --source-strong-root ROOT
  python scripts/hust_arch_suite_experiment.py evaluate --snr-root ROOT --strong-root ROOT
  python scripts/hust_arch_suite_experiment.py summarize
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

from hust_arch_suite_models import METHODS, HUSTArchitectureCandidate, paired_ce, parameter_count
from v1_noise_consistency import fix_randomness, metrics, score_checkpoints
import v2_experiment as data

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/hust_arch_suite_v1.json"
DEFAULT_OUT = Path("outputs/hust_arch_suite_v1")
SEEDS = (42, 43, 44)
TRAIN_NOISE = (10, 15, 20, 25, 30, 35, 40)
VAL_NOISE = (0, 5, 10, 20, 30)
CONDITIONS = (None, 10, 15, 20, 25, 30, 35, 40, 5, 0)
PROTOCOL = "HUST-architecture-nine-arm-v1-source-only"


def configuration() -> dict:
    c = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "epochs": 50,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0,
        "source_train_snr_db": list(TRAIN_NOISE),
        "source_val_snr_db": list(VAL_NOISE),
        "target_conditions": ["clean" if x is None else x for x in CONDITIONS],
        "target_training": False,
        "target_checkpoint_selection": False,
        "final_holdout_evaluated": False,
    }
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(f"Frozen config changed: {key}: {c.get(key)} != {value}")
    return c


def digest_parameters(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Will not overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty result rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Will not overwrite: {path}")
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def infer(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch: int = 128) -> np.ndarray:
    model.eval()
    all_predictions = []
    for start in range(0, len(x), batch):
        inputs = torch.from_numpy(x[start:start+batch]).to(device=device)
        all_predictions.append(model(inputs).argmax(dim=1).cpu().numpy())
    if not all_predictions:
        raise ValueError("Empty infer input")
    return np.concatenate(all_predictions)


def source_views(source_strong_root: Path):
    """Source-only API; materialized 0/5 files must belong to SOURCE-Val."""
    if not source_strong_root.is_dir():
        raise FileNotFoundError(source_strong_root)
    tr, tm, y = data.source_data("train")
    vr, vm, vy = data.source_data("val")
    if (len(y), len(vy)) != (738, 302):
        raise ValueError("Frozen Source-Train/Val size changed")
    train_ids, val_ids = {x["record_id"] for x in tm}, {x["record_id"] for x in vm}
    if len(train_ids) != 34 or len(val_ids) != 15 or train_ids & val_ids:
        raise ValueError("Source split record-family integrity failure")
    train_clean = data.raw_cycles(tr, tm)
    val_clean = data.raw_cycles(vr, vm)
    fixed = data.fixed_noisy_validation(
        vr, vm, noise_levels=VAL_NOISE,
        materialized_root=source_strong_root / "generated/source"
    )
    if tuple(sorted(fixed)) != tuple(sorted(VAL_NOISE)):
        raise ValueError("Missing fixed Source-Val strong-noise bands")
    for raw in [train_clean, val_clean, *fixed.values()]:
        if raw.ndim != 3 or raw.shape[1:] != (5, 4096) or not np.isfinite(raw).all():
            raise ValueError("Invalid raw cycle array")
    return (tr, tm, y, train_clean), (vr, vm, vy, val_clean, fixed)


def train(method: str, seed: int, out: Path, source_strong_root: Path) -> None:
    c = configuration()
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Only frozen methods and seeds permitted")
    dst = out / "models" / method / f"seed{seed}"
    if dst.exists():
        raise FileExistsError(f"Existing experiment would be overwritten: {dst}")
    fix_randomness(seed)
    (tr, tm, y, clean), (vr, vm, vy, val_clean, val_noise) = source_views(source_strong_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HUSTArchitectureCandidate(method).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    initial_parameters_sha256 = digest_parameters(model)
    history: list[dict] = []
    best_so_far = None
    saved_state_by_epoch: dict[int, dict] = {}

    for epoch in range(1, c["epochs"] + 1):
        # This is exactly V5's seven-level per-record/per-phase RMS Gaussian
        # schedule and epoch-specific deterministic noise random stream.
        noisy = data.fresh_noisy_source(tr, tm, seed, epoch, noise_levels=TRAIN_NOISE)
        if noisy.shape != clean.shape:
            raise ValueError("Noisy source metadata mismatch")
        perm = np.random.default_rng(seed + 901 + epoch).permutation(len(y))
        model.train()
        losses = []
        for k in range(0, len(perm), c["batch_size"]):
            ids = perm[k:k + c["batch_size"]]
            a = torch.from_numpy(clean[ids]).to(device)
            b = torch.from_numpy(noisy[ids]).to(device)
            labels = torch.from_numpy(y[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = paired_ce(model, a, b, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite train loss method={method} seed={seed} epoch={epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        clean_f1 = metrics(vy, infer(model, val_clean, device))["macro_f1"]
        noisy_f1 = {str(snr): metrics(vy, infer(model, val_noise[snr], device))["macro_f1"]
                    for snr in VAL_NOISE}
        item = dict(epoch=epoch, clean_f1=float(clean_f1), noisy_f1=noisy_f1,
                    noise_mean_f1=float(np.mean(list(noisy_f1.values()))),
                    train_loss_mean=float(np.mean(losses)))
        history.append(item)
        # We must retain all epochs until training ends: retrospective clean
        # tolerance depends on the GLOBAL max clean F1 across 50 epochs.
        saved_state_by_epoch[epoch] = {key: val.detach().cpu().clone()
                                      for key, val in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == c["epochs"]:
            print(f"{method} seed{seed} [{epoch:02d}/{c['epochs']}] "
                  f"src_clean={clean_f1:.4f} src_noise5={item['noise_mean_f1']:.4f}", flush=True)

    selected = score_checkpoints(history)
    if int(selected["epoch"]) not in saved_state_by_epoch:
        raise AssertionError("Invalid source selected epoch")
    selected_state = saved_state_by_epoch[int(selected["epoch"])]
    # Save only after a complete source-only run, with exclusive-output checks.
    dst.mkdir(parents=True, exist_ok=False)
    torch.save(dict(state=selected_state, method=method, seed=seed,
                    protocol=PROTOCOL, source_only=True,
                    final_holdout_evaluated=False, selected=selected),
               dst / "best.pt")
    write_json(dst / "history.json", history)
    write_json(dst / "run.json", dict(method=method, seed=seed, protocol=PROTOCOL,
               selected=selected, initial_parameters_sha256=initial_parameters_sha256,
               parameters=parameter_count(model), training=c,
               source_train_records=34, source_train_cycles=738,
               source_val_records=15, source_val_cycles=302,
               model_init_from_scratch=True, source_only=True,
               target_used_for_training_or_selection=False,
               final_holdout_evaluated=False))


def target_raw(snr: int | None, *, snr_root: Path | None,
               strong_root: Path | None, dev_export_root: Path | None):
    """Target-Dev ONLY: reuse existing frozen loader; never open Final-Holdout."""
    if dev_export_root is not None:
        return data.target_dev_from_export(snr, dev_export_root)
    if snr in (0, 5):
        if strong_root is None:
            raise ValueError("Generated 0/5dB Target-Dev root required")
        return data.target_dev_from_mat(snr, strong_root / "generated/target")
    return data.target_dev_from_mat(snr, snr_root)


def evaluate(method: str, seed: int, out: Path, conditions: list[int | None], *,
             snr_root: Path | None, strong_root: Path | None,
             dev_export_root: Path | None):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Unknown method/seed")
    from prepare import OUT as PREPARE_OUT, read
    frozen = read(PREPARE_OUT / "target_dev_clean_windows.json")
    if len(frozen) != 2474 or len({r["record_id"] for r in frozen}) != 88:
        raise ValueError("Frozen Target-Dev cohort mismatch")
    truth = np.asarray([r["label"] for r in frozen], dtype=np.int64)
    if not np.array_equal(np.unique(truth), np.arange(6)):
        raise ValueError("Expected all 6 target labels")
    folder = out / "models" / method / f"seed{seed}"
    record = json.loads((folder / "run.json").read_text())
    if (record["method"], record["seed"], record["protocol"]) != (method, seed, PROTOCOL):
        raise ValueError("Incorrect experiment provenance")
    if record["target_used_for_training_or_selection"] or record["final_holdout_evaluated"]:
        raise ValueError("Previous Target use detected")
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    if (saved["method"], saved["seed"], saved["protocol"]) != (method, seed, PROTOCOL):
        raise ValueError("Saved checkpoint provenance mismatch")
    fix_randomness(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HUSTArchitectureCandidate(method).to(device)
    model.load_state_dict(saved["state"], strict=True)
    if parameter_count(model) != record["parameters"]:
        raise ValueError("Architecture changed after training")
    for snr in conditions:
        label = "clean" if snr is None else f"snr{snr}"
        target = out / "evaluations" / method / f"seed{seed}" / f"target_dev_{label}.json"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        raws, metadata = target_raw(snr, snr_root=snr_root,
                                    strong_root=strong_root, dev_export_root=dev_export_root)
        if metadata != frozen:
            raise ValueError(f"Target cycle identities/order/labels differ: {label}")
        x = data.raw_cycles(raws, metadata)
        if x.shape != (2474, 5, 4096):
            raise ValueError("Incorrect Target-Dev array")
        pred = infer(model, x, device)
        score = metrics(truth, pred)
        write_json(target, dict(method=method, seed=seed, condition=label,
                   protocol=PROTOCOL, record_families=88, windows=2474,
                   truth=truth.tolist(), predicted=pred.tolist(), metrics=score,
                   selected_source=record["selected"], source_only_selection=True,
                   synthetic_target_noise=snr in (0, 5), final_holdout_evaluated=False))
        print(f"DEV {method} seed={seed} {label} Macro-F1={100*score['macro_f1']:.3f}%", flush=True)


def summarize(out: Path, methods: list[str], seeds: list[int], conditions: list[int | None]) -> None:
    per_seed: list[dict] = []
    per_class: list[dict] = []
    summary: list[dict] = []
    reference_truth = None
    for method in methods:
        for seed in seeds:
            for snr in conditions:
                cond = "clean" if snr is None else f"snr{snr}"
                path = out / "evaluations" / method / f"seed{seed}" / f"target_dev_{cond}.json"
                obj = json.loads(path.read_text())
                if (obj["method"], obj["seed"], obj["condition"], obj["protocol"]) != (method, seed, cond, PROTOCOL):
                    raise ValueError(f"Wrong provenance {path}")
                if obj["record_families"] != 88 or obj["windows"] != 2474 or obj["final_holdout_evaluated"] is not False:
                    raise ValueError("Incorrect result cohort")
                if reference_truth is None:
                    reference_truth = obj["truth"]
                if reference_truth != obj["truth"]:
                    raise ValueError("Models/conditions do not use identical target labels/order")
                m = obj["metrics"]
                per_seed.append(dict(method=method, seed=seed, condition=cond,
                    macro_f1_percent=100*m["macro_f1"], accuracy_percent=100*m["accuracy"],
                    fault_as_healthy=m["missed_fault_as_normal"],
                    healthy_as_fault=m["normal_false_alarms"]))
                for cls, (f1, recall) in enumerate(zip(m["per_class_f1"], m["per_class_recall"])):
                    per_class.append(dict(method=method, seed=seed, condition=cond,
                                          label=cls, f1_percent=100*f1,
                                          recall_percent=100*recall))
    for method in methods:
        for snr in conditions:
            cond = "clean" if snr is None else f"snr{snr}"
            rows = [z for z in per_seed if z["method"] == method and z["condition"] == cond]
            f1 = np.asarray([z["macro_f1_percent"] for z in rows])
            acc = np.asarray([z["accuracy_percent"] for z in rows])
            if len(rows) != len(seeds):
                raise ValueError("Missing seed result")
            summary.append(dict(method=method, condition=cond, seeds=len(rows),
                 macro_f1_mean=float(f1.mean()),
                 macro_f1_sd=float(f1.std(ddof=1)) if len(f1) > 1 else 0.,
                 accuracy_mean=float(acc.mean()),
                 accuracy_sd=float(acc.std(ddof=1)) if len(acc) > 1 else 0.))
    report = out / "reports"
    write_csv(report / "PER_SEED_RESULTS.csv", per_seed)
    write_csv(report / "PER_CLASS_PER_SEED.csv", per_class)
    write_csv(report / "SUMMARY.csv", summary)
    write_json(report / "SUMMARY.json", dict(protocol=PROTOCOL,
                methods=methods, seeds=seeds, conditions=["clean" if c is None else f"snr{c}" for c in conditions],
                target_dev_records=88, final_holdout_evaluated=False, summary=summary))
    print("Nine-arm report saved:", report, flush=True)


def smoke() -> None:
    from hust_arch_suite_models import smoke as models_smoke
    assert configuration()["methods"] == list(METHODS)
    models_smoke()
    rows = [dict(epoch=i, clean_f1=a, noise_mean_f1=b) for i, a, b
            in ((1, 1., .8), (2, .98, .99), (3, .995, .87))]
    assert score_checkpoints(rows)["epoch"] == 3
    print("Frozen protocol / model smoke PASS", flush=True)


def main():
    config = configuration()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "train", "evaluate", "summarize"))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--source-strong-root", type=Path)
    p.add_argument("--snr-root", type=Path)
    p.add_argument("--strong-root", type=Path)
    p.add_argument("--dev-export-root", type=Path)
    p.add_argument("--conditions", nargs="+", default=["clean", "10", "15", "20", "25", "30", "35", "40", "5", "0"])
    args = p.parse_args()
    conditions = [None if x == "clean" else int(x) for x in args.conditions]
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        p.error("Duplicate methods or seeds")
    if not args.seeds or any(z not in SEEDS for z in args.seeds):
        p.error("Seed must be in 42/43/44")
    if len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        p.error("Unexpected condition")
    if args.stage == "smoke":
        smoke()
    elif args.stage == "train":
        if args.source_strong_root is None:
            p.error("train requires --source-strong-root")
        for method in args.methods:
            for seed in args.seeds:
                train(method, seed, args.out, args.source_strong_root)
    elif args.stage == "evaluate":
        if args.snr_root is None and args.dev_export_root is None:
            p.error("evaluate needs --snr-root or --dev-export-root")
        if any(c in (0, 5) for c in conditions) and args.strong_root is None and args.dev_export_root is None:
            p.error("synthetic 0/5 dB needs --strong-root or --dev-export-root")
        for method in args.methods:
            for seed in args.seeds:
                evaluate(method, seed, args.out, conditions,
                         snr_root=args.snr_root, strong_root=args.strong_root,
                         dev_export_root=args.dev_export_root)
    else:
        summarize(args.out, args.methods, args.seeds, conditions)

if __name__ == "__main__":
    main()
