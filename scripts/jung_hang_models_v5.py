"""Paper-informed V5 adapters for Jung (TPEL 2025) and Hang (TIE 2023).

Faithfulness boundary:
  * Jung: 2-cycle, 1-cycle stride and bounded THD+N white-noise injection.
    The optional zero-current limiter is adapted to 5-phase fault labels.
  * Hang: single fundamental VSD plane and per-instant current-vector
    normalization before 40x40 trajectory rasterization.
  * Both use the established five-phase network adaptations from V3.
Original authors' code, 3L-NPC fault timing, and 3-phase PMSM topology
are NOT reproduced by this file.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from jung_hang_models_v3 import (
    JungModifiedCNN, HangWCNN, HaarDWTLLPool,
    five_phase_vsd, _interpolate_current_window, _draw_polyline,
)

# Labels are the HUST five-phase classes, not the 3L-NPC switch numbers.
FAULT_PHASES = {
    0: (), 1: (0,), 2: (0,), 3: (0,), 4: (0, 1), 5: (0, 2),
}


def forward_two_cycle_indices(meta: list[dict], tolerance: float = 0.05):
    """Construct disjoint-split *forward* 2-cycle windows, stride one cycle.

    Output pairs (anchor_metadata_index, t0, t2). No backward fallback and
    no duplication of the first window. There is no cross-record, cross-class,
    or cross-discontinuous-time pairing. Operate separately on each split.
    """
    if not 0 <= tolerance < 0.5:
        raise ValueError("invalid cycle continuity tolerance")
    groups = defaultdict(list)
    for i, row in enumerate(meta):
        groups[(row["record_id"], int(row["label"]))].append((i, row))
    windows = []
    for group in groups.values():
        group.sort(key=lambda pair: pair[1]["start"])
        for (anchor_index, a), (_, b) in zip(group[:-1], group[1:]):
            width_a = float(a["end"] - a["start"])
            width_b = float(b["end"] - b["start"])
            if min(width_a, width_b) <= 0:
                raise ValueError("nonpositive electrical period")
            if abs(a["end"] - b["start"]) <= tolerance * max(width_a, width_b):
                windows.append((int(anchor_index), float(a["start"]), float(b["end"])))
    windows.sort(key=lambda q: q[0])
    if not windows:
        raise ValueError("no valid contiguous two-cycle windows in this split")
    anchor = np.asarray([r[0] for r in windows], dtype=np.int64)
    if len(np.unique(anchor)) != len(anchor):
        raise AssertionError("two-cycle anchor repeated")
    return windows


def jung_window(time, currents, start, end, points=256):
    """Normalize per-phase midrange, then apply a common scale (five-phase adaptation)."""
    _, x = _interpolate_current_window(time, currents, start, end, points)
    midpoint = 0.5 * (x.max(axis=1, keepdims=True) + x.min(axis=1, keepdims=True))
    x = x - midpoint
    scale = np.abs(x).max()
    return (x / max(float(scale), 1e-12)).astype(np.float32)


def hang_trajectory(time, currents, start, end, *,
                    points=128, size=40, phase_angles_deg=(0, 72, 144, 216, 288),
                    quiet_ratio=0.03):
    """Eq. (6) style per-time |I_alpha_beta| normalization; single image.

    Five-phase fundamental VSD stands in for original three-phase Clarke plane.
    Input physical channel order / winding phase sequence must be verified on
    actual HUST source recordings before this is treated as a final baseline.
    """
    if not 0 <= quiet_ratio < 1:
        raise ValueError("quiet_ratio must be in [0,1)")
    _, segment = _interpolate_current_window(time, currents, start, end, points)
    comp = five_phase_vsd(segment.T, tuple(phase_angles_deg))
    alpha, beta = comp[0], comp[1]
    magnitude = np.hypot(alpha, beta)
    peak = float(magnitude.max())
    if peak <= 1e-12:
        return np.zeros((1, size, size), dtype=np.float32)
    valid = magnitude > max(peak * quiet_ratio, 1e-12)
    nx = np.divide(alpha, magnitude, out=np.zeros_like(alpha), where=valid)
    ny = np.divide(beta, magnitude, out=np.zeros_like(beta), where=valid)
    return _draw_polyline(nx, ny, size, valid=valid)[None].astype(np.float32)


def _fundamental_thd_rms(time, signal, omega, max_harmonic=30):
    """Record-wise FFT bin estimate of I1_rms and native harmonic distortion.

    This is a five-phase engineering implementation of the THD+N concept;
    it is not the author's unpublished estimation routine.
    """
    t = np.asarray(time, dtype=np.float64)
    x = np.asarray(signal, dtype=np.float64)
    w = np.asarray(omega, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 5 or len(t) != len(x):
        raise ValueError("expected aligned [T,5] currents")
    # HUST source recordings contain a prefault healthy reference segment.
    # Estimate the nominal fundamental/THD from that reference, avoiding the
    # intrinsic open-switch waveform distortion being counted as sensor noise.
    reference = (t >= 0.10) & (t <= 0.45)
    if reference.sum() >= 128:
        x, w, t = x[reference], w[reference], t[reference]
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("time must increase")
    # Cycling angular speed is supplied by the existing HUST windowing code.
    f0 = float(np.median(np.abs(w))) / (2 * np.pi)
    if not 0 < f0 < 0.5 / dt:
        raise ValueError("invalid fundamental frequency")
    frequency = np.fft.rfftfreq(len(t), dt)
    spectrum = np.fft.rfft(x - x.mean(axis=0, keepdims=True), axis=0)
    # Choose strongest measured bin near nominal f0 to reduce frequency mismatch.
    near = np.flatnonzero((frequency >= 0.8 * f0) & (frequency <= 1.2 * f0))
    if not len(near):
        raise ValueError("no measured fundamental bin")
    ix = near[np.argmax(np.abs(spectrum[near]).sum(axis=1))]
    f1 = float(frequency[ix])
    fundamental = np.sqrt(2.0) * np.abs(spectrum[ix]) / len(t)
    harmonics_sq = np.zeros(5, dtype=np.float64)
    for k in range(2, max_harmonic + 1):
        if k * f1 >= frequency[-1]:
            break
        i = int(np.argmin(np.abs(frequency - k * f1)))
        harmonics_sq += (np.sqrt(2.0) * np.abs(spectrum[i]) / len(t)) ** 2
    thd = np.sqrt(harmonics_sq) / np.maximum(fundamental, 1e-12)
    return fundamental, thd


def _fault_quiet_mask(time, currents, omega, metadata, quiet_ratio=0.03):
    """Suppress augmentation on sustained low-current intervals of faulted phases.

    Unlike the original NPC switch-state limiters, this generic five-phase
    adaptation uses the known labels and local RMS, never target labels.
    """
    t = np.asarray(time)
    x = np.asarray(currents)
    mask = np.ones_like(x, dtype=bool)
    dt = float(np.median(np.diff(t)))
    f0 = float(np.median(np.abs(omega))) / (2 * np.pi)
    width = max(3, int(round(0.03 / max(f0 * dt, 1e-9))))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width) / width
    for row in metadata:
        label = int(row["label"])
        for p in FAULT_PHASES[label]:
            sel = np.flatnonzero((t >= row["start"]) & (t < row["end"]))
            if len(sel) < 3:
                continue
            rms = np.sqrt(np.maximum(np.convolve(x[sel, p] ** 2, kernel, mode="same"), 0.0))
            reference = float(np.percentile(rms, 95))
            if reference <= 1e-10:
                mask[sel, p] = False
            else:
                mask[sel, p] = rms > quiet_ratio * reference
    return mask


def jung_thdn_augment(time, currents, omega, target_percent, rng, *,
                      metadata=(), limiter="fault_quiet", quiet_ratio=0.03):
    """Target THD+N (%) using I_noise=I1*sqrt(max(target^2-THD^2,0)).

    Noise power is measured after the optional fault-region limiter, so the
    final record-wise noise RMS matches the requested *added* noise amplitude.
    A target below native THD yields zero augmentation, by construction.
    """
    if not 0 <= float(target_percent) <= 100:
        raise ValueError("THD+N percent must be between 0 and 100")
    if limiter not in {"fault_quiet", "none"}:
        raise ValueError("unknown limiter")
    clean = np.asarray(currents, dtype=np.float64)
    fundamental, native_thd = _fundamental_thd_rms(time, clean, omega)
    desired = float(target_percent) / 100.0
    sigma = fundamental * np.sqrt(np.maximum(desired ** 2 - native_thd ** 2, 0.0))
    random = rng.standard_normal(clean.shape)
    if limiter == "fault_quiet":
        random *= _fault_quiet_mask(time, clean, omega, metadata, quiet_ratio)
    measured = np.sqrt(np.mean(random ** 2, axis=0))
    multiplier = np.divide(sigma, measured, out=np.zeros_like(sigma), where=measured > 1e-12)
    noisy = clean + random * multiplier[None, :]
    if not np.isfinite(noisy).all():
        raise FloatingPointError("non-finite THD+N augmentation")
    return noisy
