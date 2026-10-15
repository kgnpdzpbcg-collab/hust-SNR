"""Export ONLY the frozen HUST five-phase IM-AA target Dev records as small NPZ files.

This is a data-transfer / analysis utility, not a training or evaluation script.
It does not touch source training, target holdout, checkpoints, or final metrics.

Input MAT schema:
  clean:  'name',   shape (10031, 7): [time, five currents, omega]
  noisy:  'I_noisy', shape (10031, 7): [time, five currents, omega]
Duplicate time stamps are merged by averaging all six measured channels,
exactly like scripts/prepare.py and scripts/v1_noise_consistency.py.

Run: python scripts/export_snr_data.py --snr-root /path/to/SNR_parent
Help: python scripts/export_snr_data.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SNR_LEVELS = (10, 15, 20, 25, 30, 35, 40)
FAULT_NAMES = {
    0: "Healthy", 1: "OUS(A)", 2: "OLS(A)", 3: "OP(A)",
    4: "OUS(A)&OUS(B)", 5: "OLS(A)&OP(C)",
}


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def select_dev(split: dict, labels: dict, expected_dev_count: int = 88) -> list[str]:
    """Strict allowlist: never select from the target holdout or its duplicates."""
    dev = split["target_dev_records"]
    holdout = set(split["target_holdout_records"])
    canonical = set(split["target_canonical_records"])
    if (len(dev) != expected_dev_count or len(dev) != len(set(dev))
            or set(dev) & holdout or not set(dev).issubset(canonical)):
        raise ValueError("Frozen Dev split invalid; refusing to export")
    if not set(dev).issubset(labels["experimental"]):
        raise ValueError("Missing experimental labels for frozen Dev records")
    for group in split.get("target_duplicate_groups", []):
        if len(set(group) & set(dev)) and len(set(group) & holdout):
            raise ValueError("Duplicate group crosses Dev and Final-Holdout")
    return sorted(dev)


def resolve_noisy(root: Path, snr: int, rid: str) -> Path:
    """Match paths accepted by V1; ambiguous matches are not allowed."""
    locations = [root / f"SNR_{snr}" / f"{rid}.mat",
                 root / f"snr{snr}" / f"{rid}.mat",
                 root / str(snr) / f"{rid}.mat"]
    found = [p for p in locations if p.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one SNR_{snr} file for {rid}; found {found}; checked {locations}")
    return found[0]


def load_raw(path: Path, field: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = loadmat(path, variable_names=[field]).get(field)
    if (not isinstance(values, np.ndarray) or values.shape != (10031, 7)
            or values.dtype != np.float64 or not np.isfinite(values).all()):
        raise ValueError(f"Unexpected MAT matrix {field} in {path}")
    if np.any(np.diff(values[:, 0]) < 0):
        raise ValueError(f"Nonmonotonic timestamps: {path}")
    if np.any(values[:, 0] < -1e-8):
        raise ValueError(f"Invalid negative timestamps: {path}")
    return values


def merge_repeated_time(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same deterministic duplicate-timestamp averaging as Gate0/V1."""
    time, starts, counts = np.unique(values[:, 0], return_index=True, return_counts=True)
    merged = np.add.reduceat(values[:, 1:], starts, axis=0) / counts[:, None]
    if len(time) != 10011 or not np.all(np.diff(time) > 0):
        raise ValueError(f"Unexpected unique time dimension: {len(time)}, expected 10011")
    return time, merged[:, :5], merged[:, -1]


def assert_paired(clean: np.ndarray, noisy: np.ndarray, snr: int, rid: str) -> float:
    """Check original 64-bit MAT fields *before* lossy float32 export."""
    if not (np.array_equal(clean[:, 0], noisy[:, 0])
            and np.array_equal(clean[:, -1], noisy[:, -1])):
        raise ValueError(f"Time/omega mismatch for record {rid} at {snr} dB")
    power = np.mean(clean[:, 1:6] ** 2, axis=0)
    noise_power = np.mean((noisy[:, 1:6] - clean[:, 1:6]) ** 2, axis=0)
    if np.any(power <= 0) or np.any(noise_power <= 0):
        raise ValueError(f"Invalid phase/noise power: {rid}, {snr} dB")
    measured = 10 * np.log10(power / noise_power)
    deviation = float(np.max(np.abs(measured - snr)))
    if deviation > 1e-6:
        raise ValueError(f"SNR mismatch: {rid}, expected {snr}, measured {measured}")
    return deviation


def validate_windows(windows: list[dict], ids: list[str], labels: dict) -> dict:
    allowed = set(ids)
    encountered = set()
    counts = {rid: 0 for rid in ids}
    for row in windows:
        rid = row["record_id"]
        if rid not in allowed or not (0 <= row["label"] <= 5):
            raise ValueError("Window list is not the frozen target Dev window list")
        expected = 0 if row["end"] <= 0.5 else labels["experimental"][rid]["fault_label"]
        if (row["label"] != expected or not 0.0 <= row["start"] < row["end"] <= 1.002
                or row["start"] < 0.5 < row["end"]):
            raise ValueError(f"Invalid window label or boundary: {rid}")
        counts[rid] += 1
        encountered.add(rid)
    if encountered != allowed or len(windows) != 2474:
        raise ValueError("Expected exactly 2474 V1 Dev windows from all 88 records")
    return {"total_windows": len(windows), "windows_by_record": counts}


def collect_reference(root: Path, clean_root: Path, ids: list[str], labels: dict):
    """Only Dev Clean is read, to provide common time/omega metadata."""
    times, omegas, faults, speeds, torques = [], [], [], [], []
    clean_cache = {}
    for rid in ids:
        values = load_raw(clean_root / f"{rid}.mat", "name")
        t, current, omega = merge_repeated_time(values)
        times.append(t)
        omegas.append(omega)
        clean_cache[rid] = values
        entry = labels["experimental"][rid]
        faults.append(int(entry["fault_label"]))
        speeds.append(float(entry["speed"]))
        torques.append(float(entry["torque"]))
    return {
        "time": np.stack(times), "omega": np.stack(omegas),
        "fault_label": np.array(faults, dtype=np.int8),
        "speed": np.array(speeds, dtype=np.float64),
        "torque": np.array(torques, dtype=np.float64),
        "record_ids": np.array(ids),
    }, clean_cache


def collect_condition(ids: list[str], reference: dict, clean_cache: dict,
                      snr_root: Path | None, snr: int | None):
    """Export unnormalized, time-deduplicated five-phase currents [N,10011,5]."""
    currents = []
    max_snr_error = 0.0
    for index, rid in enumerate(ids):
        clean = clean_cache[rid]
        if snr is None:
            raw = clean
        else:
            raw = load_raw(resolve_noisy(snr_root, snr, rid), "I_noisy")
            max_snr_error = max(max_snr_error, assert_paired(clean, raw, snr, rid))
        t, current, omega = merge_repeated_time(raw)
        if not (np.array_equal(t, reference["time"][index])
                and np.allclose(omega, reference["omega"][index], rtol=0, atol=1e-12)):
            raise ValueError(f"Clean/Noisy merged time or omega mismatch: {rid}")
        currents.append(current.astype(np.float32))
    return np.stack(currents), max_snr_error


def write_chunks(folder: Path, stem: str, arrays: np.ndarray, chunk_size: int) -> list[dict]:
    """Chunk optional; every part saves explicit *global* record indices."""
    index = np.arange(len(arrays), dtype=np.int32)
    outputs = []
    n_parts = (len(arrays) + chunk_size - 1) // chunk_size
    for part, start in enumerate(range(0, len(arrays), chunk_size), 1):
        end = min(len(arrays), start + chunk_size)
        suffix = "" if n_parts == 1 else f"_part{part:02d}"
        path = folder / f"dev_{stem}{suffix}.npz"
        np.savez_compressed(path, currents=arrays[start:end], record_indices=index[start:end])
        with np.load(path, allow_pickle=False) as check:
            if not (np.array_equal(check["currents"], arrays[start:end])
                    and np.array_equal(check["record_indices"], index[start:end])):
                raise RuntimeError(f"NPZ round-trip mismatch: {path}")
        outputs.append({"file": path.name, "records": int(end - start), "index_range": [start, end],
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
    return outputs


def export_dataset(repo_root: Path, snr_root: Path, output: Path,
                   records_per_file: int = 88, include_clean: bool = True) -> dict:
    """Export only the canonical 88-record Dev; never overwrite or read Holdout."""
    repo_root = repo_root.resolve()
    snr_root = snr_root.resolve()
    output = output.resolve()
    if not 1 <= records_per_file <= 88:
        raise ValueError("--records-per-file must be between 1 and 88")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    split_path = repo_root / "configs" / "splits_gate0_v1.json"
    labels_path = repo_root / "analysis" / "data_audit_v1" / "LABELS.json"
    windows_path = repo_root / "outputs" / "gate0_v1" / "target_dev_clean_windows.json"
    split = read_json(split_path)
    labels = read_json(labels_path)
    ids = select_dev(split, labels)
    windows = read_json(windows_path)
    window_stats = validate_windows(windows, ids, labels)
    clean_root = repo_root / "data" / "target_dev" / "clean"
    if not clean_root.is_dir():
        raise FileNotFoundError(f"Clean Dev root missing: {clean_root}")
    # Avoid any accidental export into source-controlled code and analysis directories.
    try:
        output.relative_to(repo_root / "outputs")
    except ValueError as exc:
        raise ValueError("Output must be under <repo>/outputs/ (gitignored)") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=".snr_export_", dir=output.parent))
    try:
        reference, clean_cache = collect_reference(repo_root, clean_root, ids, labels)
        reference_file = folder / "dev_reference.npz"
        np.savez_compressed(reference_file, **reference)
        with np.load(reference_file, allow_pickle=False) as check:
            for key, value in reference.items():
                if not np.array_equal(check[key], value):
                    raise RuntimeError(f"Reference round-trip mismatch: {key}")
        (folder / "dev_cycle_windows.json").write_text(
            json.dumps(windows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        exported = {}
        for snr in ([None] if include_clean else []) + list(SNR_LEVELS):
            stem = "clean" if snr is None else f"snr{snr}"
            array, deviation = collect_condition(ids, reference, clean_cache, snr_root, snr)
            chunks = write_chunks(folder, stem, array, records_per_file)
            exported[stem] = {"shape": list(array.shape), "dtype": "float32",
                              "max_original_snr_error_db": deviation if snr is not None else None,
                              "files": chunks}
            print(f"{stem}: {len(ids)} Dev records; {len(chunks)} archive(s); "
                  f"size={sum(x['bytes'] for x in chunks)/1024**2:.2f} MiB", flush=True)
        files = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.name != "manifest.json":
                files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "status": "dev_only_export_completed", "dataset": "HUST Five-phase IM-AA",
            "scope": "88 frozen unique target Dev records, never Final-Holdout",
            "final_holdout_evaluated_or_exported": False,
            "source_split_sha256": sha256(split_path), "source_labels_sha256": sha256(labels_path),
            "source_cycle_windows_sha256": sha256(windows_path),
            "record_ids_order": ids,
            "source_record_count": len(ids), "final_holdout_record_count_not_exported": len(split["target_holdout_records"]),
            "record_metadata": [
                {"record_id": rid, "fault_label_after_0_5s": int(reference["fault_label"][i]),
                 "fault_name": FAULT_NAMES[int(reference["fault_label"][i])],
                 "speed": float(reference["speed"][i]), "torque": float(reference["torque"][i])}
                for i, rid in enumerate(ids)],
            "cycle_window_stats": window_stats,
            "file_schema": {
                "dev_reference.npz": "record_ids(N),time(N,10011) float64,omega(N,10011) float64, fault_label(N), speed(N), torque(N)",
                "dev_*.npz": "currents(K,10011,5) float32 UNNORMALIZED; record_indices(K) refers to global record_ids_order",
                "dev_cycle_windows.json": "unaltered frozen V1 Dev window metadata (record_id/start/end/label)"},
            "processing": "duplicate timestamp averaged before export; raw current kept in physical units; no 128 resampling/normalization; float64 -> float32 only for current",
            "snr_definition": "per phase, original 10031-point full record, before time merging: 10 log10(mean(clean_current^2)/mean((noisy-clean)^2))",
            "raw_noise_distribution": "unknown; no distribution inferred",
            "conditions": exported, "file_sha256": files,
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(folder, output)
        print(f"EXPORT COMPLETE: {output} / manifest.json", flush=True)
    finally:
        if folder.exists():
            shutil.rmtree(folder)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1],
                   help="Repository root containing frozen configs, labels, Clean Dev and window metadata")
    p.add_argument("--snr-root", type=Path, required=True,
                   help="Parent directory containing SNR_10 ... SNR_40 subdirectories")
    p.add_argument("--output", type=Path, help="New directory under <repo>/outputs/, default outputs/dev_snr_export")
    p.add_argument("--records-per-file", type=int, default=88,
                   help="88 -> one NPZ per SNR; 20 -> <=20 records per part (~4 MB of current data)")
    p.add_argument("--no-clean", action="store_true", help="Do not export optional Clean-current NPZ")
    args = p.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output if args.output else repo_root / "outputs" / "dev_snr_export"
    export_dataset(repo_root, args.snr_root, output, args.records_per_file, not args.no_clean)


if __name__ == "__main__":
    main()
