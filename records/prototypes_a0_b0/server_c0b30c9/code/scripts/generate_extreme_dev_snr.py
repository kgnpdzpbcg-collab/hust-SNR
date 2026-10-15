"""Generate -5/-10 dB measurement stress sets for ONLY frozen Target-Dev IDs.

Source signals are paired author Clean(name) and 10dB(I_noisy) MAT files.
Output retains the original 10031x7 schema/timestamps/angular velocity.
Do not treat scaled noise as independent experimental measurements.
No Final-Holdout MAT files are read or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

LEVELS = (-5, -10)


def full_record_phase_snr(clean: np.ndarray, current: np.ndarray) -> np.ndarray:
    """SNR is defined over the WHOLE record, per phase, not per fault window."""
    if clean.shape != current.shape or clean.ndim != 2 or clean.shape[1] != 5:
        raise ValueError("Expected paired [time,5] phase currents")
    power = np.mean(clean.astype(np.float64) ** 2, axis=0)
    noise_power = np.mean((current - clean).astype(np.float64) ** 2, axis=0)
    if np.any(power <= 0) or np.any(noise_power <= 0):
        raise ValueError("Zero power in SNR input")
    return 10 * np.log10(power / noise_power)


def scale_author_residual(clean: np.ndarray, reference10: np.ndarray,
                          level: int) -> np.ndarray:
    if level not in LEVELS:
        raise ValueError("Only -5/-10 supported in this generator")
    measured = full_record_phase_snr(clean, reference10)
    if np.max(np.abs(measured - 10.0)) > 1e-6:
        raise ValueError(f"Input author SNR10 misaligned: {measured}")
    factor = 10 ** ((10 - level) / 20)
    return clean + factor * (reference10 - clean)


def _matrix(path: Path, field: str) -> np.ndarray:
    x = loadmat(path, variable_names=[field])[field]
    if x.shape != (10031, 7) or x.dtype != np.float64 or not np.isfinite(x).all():
        raise ValueError(f"Unexpected MAT field/shape/finite: {path}/{field}")
    return x


def generate(clean_root: Path, snr10_root: Path, output_root: Path) -> None:
    from prepare import OUT as PREPARE_OUT, read
    manifest = read(PREPARE_OUT / "target_dev_clean_windows.json")
    ids = sorted({entry["record_id"] for entry in manifest})
    if len(manifest) != 2474 or len(ids) != 88:
        raise ValueError("Frozen Target-Dev index must contain exactly 88 record families")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output_root}")
    # Validate ALL needed inputs before creating any new output directory.
    pairs = []
    for rid in ids:
        clean_path = clean_root / f"{rid}.mat"
        noisy_path = snr10_root / f"{rid}.mat"
        if not clean_path.is_file() or not noisy_path.is_file():
            raise FileNotFoundError(f"Missing Clean/10dB pair: {rid}")
        clean = _matrix(clean_path, "name")
        reference = _matrix(noisy_path, "I_noisy")
        if not np.array_equal(clean[:, [0, 6]], reference[:, [0, 6]]):
            raise ValueError(f"Time/angular velocity mismatch: {rid}")
        if np.any(np.diff(clean[:, 0]) < 0):
            raise ValueError(f"Nonmonotonic timestamps: {rid}")
        if np.max(np.abs(full_record_phase_snr(
                clean[:, 1:6], reference[:, 1:6]) - 10.0)) > 1e-6:
            raise ValueError(f"Reference SNR10 failed verification: {rid}")
        pairs.append((rid, clean, reference))
    output_root.mkdir(parents=True, exist_ok=False)
    for level in LEVELS:
        (output_root / f"SNR_{level}").mkdir()
    files = []
    for rid, clean, reference in pairs:
        for level in LEVELS:
            result = clean.copy()
            result[:, 1:6] = scale_author_residual(
                clean[:, 1:6], reference[:, 1:6], level)
            measured = full_record_phase_snr(clean[:, 1:6], result[:, 1:6])
            if np.max(np.abs(measured - level)) > 1e-6:
                raise AssertionError("Extreme SNR accuracy mismatch")
            path = output_root / f"SNR_{level}" / f"{rid}.mat"
            savemat(path, {"I_noisy": result}, do_compression=True)
            saved = _matrix(path, "I_noisy")
            if not np.array_equal(saved, result):
                raise AssertionError(f"Export roundtrip failed: {path}")
            checksum = hashlib.sha256(np.ascontiguousarray(saved).tobytes()).hexdigest()
            files.append({"file": str(path.relative_to(output_root)),
                          "record_id": rid, "snr_db": level,
                          "snr_by_phase": measured.tolist(),
                          "matrix_sha256": checksum})
    (output_root / "GENERATION_MANIFEST.json").write_text(
        json.dumps({
            "source": "author_10dB_residual_scaled",
            "levels": list(LEVELS),
            "dev_family_count": 88,
            "file_count": len(files),
            "synthetic_stress_only": True,
            "independent_measurements": False,
            "final_holdout_accessed": False,
            "models_trained_or_evaluated": False,
            "files": files,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DEV extreme SNR generation complete: {len(files)} MAT files")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-root", type=Path, required=True,
                   help="Original HUST Clean measurement folder with id.mat files")
    p.add_argument("--snr10-root", type=Path, required=True,
                   help="Original HUST published SNR10 measurement folder")
    p.add_argument("--out", type=Path, required=True,
                   help="NEW empty/nonexistent directory for SNR_-5/SNR_-10")
    a = p.parse_args()
    generate(a.clean_root, a.snr10_root, a.out)


if __name__ == "__main__":
    main()
