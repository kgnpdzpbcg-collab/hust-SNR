"""V4 paper-informed Jung/Hang input adapters (no changes to historical V2/V3).
This remains a five-phase adaptation, not an exact author implementation.
"""
from __future__ import annotations

import numpy as np

from jung_hang_models_v3 import (
    JungModifiedCNN, HangWCNN, HaarDWTLLPool, five_phase_vsd,
    _interpolate_current_window, _draw_polyline,
    add_white_noise_snr_preserve_quiet,
)


def jung_input(time, currents, start, end, points=128, scale_mode="shared"):
    """Preserve five-phase polarity; compare per-phase and shared normalization."""
    _, segment = _interpolate_current_window(time, currents, start, end, points)
    centered = segment - 0.5 * (segment.max(axis=1) + segment.min(axis=1))[:, None]
    if scale_mode == "shared":
        divisor = np.full((5, 1), np.max(np.abs(centered)), dtype=np.float64)
    elif scale_mode == "per_phase":
        divisor = np.max(np.abs(centered), axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown Jung scale_mode: {scale_mode}")
    out = np.divide(centered, divisor, out=np.zeros_like(centered), where=divisor > 1e-12)
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite Jung input")
    return out.astype(np.float32)


def hang_vsd_image(time, currents, start, end, points=128, image_size=40,
                   planes="dual", center_mode="none", normalization="shared",
                   quiet_ratio=0.03):
    """Two-channel fixed-capacity VSD image; zero the second channel for H1.
    
    All five phases are projected before any per-plane normalization.
    'none' preserves fault-related phase asymmetry/DC offsets.
    Shared normalization preserves radial excursions, including weak fault arcs.
    In instantaneous mode, near-zero vectors are masked relative to window peak;
    adjacency-aware rasterisation never connects through masked time points.
    """
    if planes not in {"fundamental", "dual"}:
        raise ValueError("planes must be fundamental or dual")
    if center_mode not in {"none", "phase_mean"}:
        raise ValueError("center_mode must be none or phase_mean")
    if normalization not in {"shared", "instantaneous"}:
        raise ValueError("normalization must be shared or instantaneous")
    if not 0.0 <= quiet_ratio < 1.0:
        raise ValueError("quiet_ratio outside [0,1)")
    _, segment = _interpolate_current_window(time, currents, start, end, points)
    if center_mode == "phase_mean":
        segment = segment - segment.mean(axis=1, keepdims=True)
    comp = five_phase_vsd(segment.T)
    fundamental = np.hypot(comp[0], comp[1])
    harmonic = np.hypot(comp[2], comp[3])
    base_scale = float(np.max(fundamental))
    if base_scale <= 1e-12:
        return np.zeros((2, image_size, image_size), dtype=np.float32)
    if normalization == "shared":
        # The same scale applies to both planes; second plane is not amplified.
        scale = max(base_scale, float(np.max(harmonic)), 1e-12)
        coords = [(comp[0] / scale, comp[1] / scale, fundamental),
                  (comp[2] / scale, comp[3] / scale, harmonic)]
    else:
        coords = []
        for a, b, mag in ((comp[0], comp[1], fundamental),
                          (comp[2], comp[3], harmonic)):
            x = np.divide(a, mag, out=np.zeros_like(a), where=mag > 1e-12)
            y = np.divide(b, mag, out=np.zeros_like(b), where=mag > 1e-12)
            coords.append((x, y, mag))
    images = []
    for idx, (x, y, mag) in enumerate(coords):
        if idx == 1 and planes == "fundamental":
            images.append(np.zeros((image_size, image_size), dtype=np.float32))
            continue
        peak = float(np.max(mag))
        # In shared mode keep even weak but nonzero fault arcs.
        threshold = (quiet_ratio * peak) if normalization == "instantaneous" else 1e-12
        valid = np.isfinite(mag) & (mag > max(threshold, 1e-12))
        images.append(_draw_polyline(x, y, image_size, valid=valid))
    result = np.stack(images).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite Hang raster")
    return result


def add_noise(time, currents, omega, snr_db, rng, mode="awgn"):
    """Exact per-record per-phase SNR AWGN, with optional V3 quiet interval gate."""
    if mode == "quiet":
        return add_white_noise_snr_preserve_quiet(time, currents, omega, snr_db, rng)
    if mode != "awgn":
        raise ValueError(f"Unknown noise mode: {mode}")
    signal = np.asarray(currents, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 5:
        raise ValueError("Expected [time,5] currents")
    if not np.isfinite(signal).all():
        raise ValueError("Non-finite source currents")
    noise = rng.standard_normal(signal.shape)
    power = np.mean(signal ** 2, axis=0, keepdims=True)
    noise_power = np.mean(noise ** 2, axis=0, keepdims=True)
    scale = np.sqrt(np.divide(power * 10 ** (-float(snr_db) / 10),
                              noise_power, out=np.zeros_like(power),
                              where=(power > 1e-18) & (noise_power > 1e-18)))
    return signal + noise * scale
