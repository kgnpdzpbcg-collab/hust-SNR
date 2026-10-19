"""Versioned Jung/Hang model adaptations for the HUST five-phase IM-AA task.

V3 preserves inter-phase amplitude ratios for Jung and uses an instantaneous-
normalized fundamental VSD trajectory with the paper-sized 40x40 WCNN input
for Hang. These are paper-informed task adaptations, not author-released code.
The legacy dual-plane trajectory helper remains available for ablation only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock1D(nn.Module):
    """Two 3-tap Conv1D layers with an identity skip, as described by Jung et al."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + residual)


class JungModifiedCNN(nn.Module):
    """Jung-inspired compact 1-D CNN for five-phase, 128-point current windows.

    The paper does not disclose every layer hyperparameter. The reconstruction
    uses a 3-tap dilated convolution (dilation=3), max pooling, a two-layer
    residual block, global average pooling, and a six-class linear head.
    White-noise augmentation belongs to the training pipeline, not this module.
    """
    def __init__(self, in_channels: int = 5, num_classes: int = 6, channels: int = 32):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            in_channels, channels, kernel_size=3, dilation=3, padding=3, bias=True
        )
        self.bn = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual = ResidualBlock1D(channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"JungModifiedCNN expects [B,C,T], got {tuple(x.shape)}")
        x = self.activation(self.bn(self.dilated_conv(x)))
        x = self.pool(x)
        x = self.residual(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


class HaarDWTLLPool(nn.Module):
    """Fixed 2-D Haar low-low subband pooling; high-frequency bands are discarded.

    The all-ones 2x2 LL filter follows the unnormalised Haar equations printed
    in Hang et al. It has no trainable parameters and halves both spatial axes.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("ll_kernel", torch.ones(1, 1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"HaarDWTLLPool expects [B,C,H,W], got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            raise ValueError(f"DWT pooling requires even H/W, got {(height, width)}")
        weight = self.ll_kernel.to(dtype=x.dtype).expand(x.shape[1], 1, 2, 2)
        return F.conv2d(x, weight, stride=2, padding=0, groups=x.shape[1])


class HangWCNN(nn.Module):
    """Five-phase adaptation of Hang's five-convolution WCNN.

    The architecture follows the paper's I-C-W-C-W-C-C-C-F-O sequence with
    channels 4/8/16/32/64. The original paper's feature-map dimensions imply a
    40x40 image: two stride-2 LL pooling stages reduce 40x40 to 10x10 before
    the final convolution and fully connected classifier. The original input is
    a single normalized Clarke trajectory; V3 uses one normalized fundamental
    VSD trajectory channel rather than an independently scaled harmonic plane.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        image_size: int = 40,
        channels: tuple[int, ...] = (4, 8, 16, 32, 64),
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("HangWCNN requires exactly five convolutional stages")
        if image_size % 4:
            raise ValueError("image_size must be divisible by 4 for two DWT pools")
        ins = (in_channels, *channels[:-1])
        self.convs = nn.ModuleList([
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=True)
            for cin, cout in zip(ins, channels)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c) for c in channels])
        self.pool1 = HaarDWTLLPool()
        self.pool2 = HaarDWTLLPool()
        self.activation = nn.ReLU(inplace=False)
        spatial = image_size // 4
        self.expected_image_size = image_size
        self.classifier = nn.Linear(channels[-1] * spatial * spatial, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"HangWCNN expects [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[-2:] != (self.expected_image_size, self.expected_image_size):
            raise ValueError(
                f"Expected {self.expected_image_size}x{self.expected_image_size} images, "
                f"got {tuple(x.shape[-2:])}"
            )
        for idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = self.activation(bn(conv(x)))
            if idx == 0:
                x = self.pool1(x)
            elif idx == 1:
                x = self.pool2(x)
        return self.classifier(torch.flatten(x, start_dim=1))


def five_phase_vsd(
    currents: np.ndarray,
    phase_angles_deg: tuple[float, ...] = (0.0, 72.0, 144.0, 216.0, 288.0),
) -> np.ndarray:
    """Return [alpha1, beta1, alpha3, beta3] for [T,5] phase currents.

    Default angles assume a uniformly spaced five-phase winding. Pass the
    verified physical phase-axis angles if the dataset's phase order differs.
    This is the standard 2/5 Fourier/VSD projection for the fundamental and
    third-harmonic plane (the latter is equivalent to the second plane for m=5).
    """
    signal = np.asarray(currents, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 5:
        raise ValueError(f"five_phase_vsd expects [T,5], got {signal.shape}")
    if len(phase_angles_deg) != 5:
        raise ValueError("phase_angles_deg must contain five phase-axis angles")
    theta = np.deg2rad(np.asarray(phase_angles_deg, dtype=np.float64))
    alpha1 = 0.4 * (signal @ np.cos(theta))
    beta1 = 0.4 * (signal @ np.sin(theta))
    alpha3 = 0.4 * (signal @ np.cos(3.0 * theta))
    beta3 = 0.4 * (signal @ np.sin(3.0 * theta))
    return np.stack((alpha1, beta1, alpha3, beta3), axis=0)


def _draw_polyline(x: np.ndarray, y: np.ndarray, size: int) -> np.ndarray:
    """Rasterise a fixed-scale 2-D trajectory without axes, labels, or plotting deps."""
    image = np.zeros((size, size), dtype=np.float32)
    px = np.rint((np.clip(x, -1.0, 1.0) + 1.0) * 0.5 * (size - 1)).astype(int)
    py = np.rint((1.0 - (np.clip(y, -1.0, 1.0) + 1.0) * 0.5) * (size - 1)).astype(int)
    if len(px) == 1:
        image[py[0], px[0]] = 1.0
        return image
    for x0, y0, x1, y1 in zip(px[:-1], py[:-1], px[1:], py[1:]):
        steps = max(abs(int(x1 - x0)), abs(int(y1 - y0)), 1) * 2
        xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
        ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
        image[ys, xs] = 1.0
    return image


def current_window_to_vsd_images(
    time: np.ndarray,
    currents: np.ndarray,
    start: float,
    end: float,
    points: int = 128,
    image_size: int = 64,
    phase_angles_deg: tuple[float, ...] = (0.0, 72.0, 144.0, 216.0, 288.0),
) -> np.ndarray:
    """Convert one raw five-phase current window to [2,H,W] trajectory images.

    Channel 0 is the alpha1-beta1 plane; channel 1 is alpha3-beta3. DC offsets
    are removed by per-phase mean-centering, while all four VSD components share
    one amplitude scale so the weak harmonic-plane magnitude is not amplified alone.
    Axes and plot decorations are omitted. A near-zero window returns blank maps.
    """
    t = np.asarray(time, dtype=np.float64)
    current = np.asarray(currents, dtype=np.float64)
    if current.ndim != 2 or current.shape[1] != 5:
        raise ValueError(f"currents must be [T,5], got {current.shape}")
    if t.ndim != 1 or len(t) != len(current) or not start < end:
        raise ValueError("Invalid time/current shape or window bounds")
    grid = np.linspace(start, end, points, dtype=np.float64)
    segment = np.stack([np.interp(grid, t, current[:, p]) for p in range(5)], axis=1)
    centered = segment - segment.mean(axis=0, keepdims=True)
    components = five_phase_vsd(centered, phase_angles_deg)
    scale = float(np.max(np.abs(components)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros((2, image_size, image_size), dtype=np.float32)
    components = np.clip(components / scale, -1.0, 1.0)
    image1 = (
        _draw_polyline(components[0], components[1], image_size)
        if float(np.max(np.abs(components[:2]))) > 1e-6
        else np.zeros((image_size, image_size), dtype=np.float32)
    )
    image3 = (
        _draw_polyline(components[2], components[3], image_size)
        if float(np.max(np.abs(components[2:]))) > 1e-6
        else np.zeros((image_size, image_size), dtype=np.float32)
    )
    return np.stack((image1, image3), axis=0).astype(np.float32)


def _interpolate_current_window(
    time: np.ndarray,
    currents: np.ndarray,
    start: float,
    end: float,
    points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a raw window and interpolate it on a fixed grid."""
    t = np.asarray(time, dtype=np.float64)
    current = np.asarray(currents, dtype=np.float64)
    if current.ndim != 2 or current.shape[1] != 5:
        raise ValueError(f"currents must be [T,5], got {current.shape}")
    if t.ndim != 1 or len(t) != len(current) or len(t) < 2:
        raise ValueError("time/current dimensions are inconsistent")
    if not np.isfinite(t).all() or not np.isfinite(current).all():
        raise ValueError("time/current contain non-finite values")
    if not np.all(np.diff(t) > 0) or not t[0] <= start < end <= t[-1]:
        raise ValueError("time must be strictly increasing and contain the requested window")
    if points < 8:
        raise ValueError("points must be at least 8")
    grid = np.linspace(start, end, points, dtype=np.float64)
    segment = np.stack(
        [np.interp(grid, t, current[:, phase]) for phase in range(5)], axis=0
    )
    return grid, segment


def current_window_to_jung_input(
    time: np.ndarray,
    currents: np.ndarray,
    start: float,
    end: float,
    points: int = 128,
) -> np.ndarray:
    """Build [5,T] Jung input using a phase-shared scale.

    Each phase is midpoint-centered, but all five phases share one scale.
    This removes a common window amplitude factor without independently
    normalizing each phase and erasing fault-related inter-phase amplitude ratios.
    This is an explicit five-phase task adaptation; the original three-phase
    paper does not prescribe this exact normalization.
    """
    _, segment = _interpolate_current_window(time, currents, start, end, points)
    midpoint = (segment.max(axis=1) + segment.min(axis=1)) * 0.5
    centered = segment - midpoint[:, None]
    scale = float(np.max(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros((5, points), dtype=np.float32)
    result = centered / scale
    if not np.isfinite(result).all():
        raise FloatingPointError("Jung input normalization produced non-finite values")
    return result.astype(np.float32)


def _draw_polyline(
    x: np.ndarray,
    y: np.ndarray,
    size: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize valid adjacent trajectory points without axes or plot decorations."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or size < 8:
        raise ValueError("invalid trajectory coordinates or image size")
    if valid is None:
        valid = np.isfinite(x) & np.isfinite(y)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(x) & np.isfinite(y)
        if valid.shape != x.shape:
            raise ValueError("valid mask must match trajectory coordinates")
    image = np.zeros((size, size), dtype=np.float32)
    if not valid.any():
        return image
    px = np.rint((np.clip(x, -1.0, 1.0) + 1.0) * 0.5 * (size - 1)).astype(int)
    py = np.rint((1.0 - (np.clip(y, -1.0, 1.0) + 1.0) * 0.5) * (size - 1)).astype(int)
    image[py[valid], px[valid]] = 1.0
    for idx in range(len(px) - 1):
        if not (valid[idx] and valid[idx + 1]):
            continue
        x0, y0, x1, y1 = px[idx], py[idx], px[idx + 1], py[idx + 1]
        steps = max(abs(int(x1 - x0)), abs(int(y1 - y0)), 1) * 2
        xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
        ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
        image[ys, xs] = 1.0
    return image


def current_window_to_normalized_vsd_image(
    time: np.ndarray,
    currents: np.ndarray,
    start: float,
    end: float,
    points: int = 128,
    image_size: int = 40,
    phase_angles_deg: tuple[float, ...] = (0.0, 72.0, 144.0, 216.0, 288.0),
    eps: float = 1e-8,
) -> np.ndarray:
    """Build Hang's single-channel [1,40,40] normalized current-vector image.

    Five-phase adaptation of the paper's three-phase Clarke trajectory:
    project into the fundamental VSD plane, remove per-phase DC offset, and
    normalize the alpha-beta vector magnitude independently at each time step.
    Near-zero vectors are omitted instead of being amplified into arbitrary
    directions. A shared fixed coordinate range is then rasterized.
    """
    _, segment = _interpolate_current_window(time, currents, start, end, points)
    centered = segment - segment.mean(axis=1, keepdims=True)
    components = five_phase_vsd(centered.T, phase_angles_deg)
    alpha, beta = components[0], components[1]
    magnitude = np.hypot(alpha, beta)
    valid = np.isfinite(magnitude) & (magnitude > eps)
    x = np.zeros_like(alpha)
    y = np.zeros_like(beta)
    x[valid] = alpha[valid] / magnitude[valid]
    y[valid] = beta[valid] / magnitude[valid]
    image = _draw_polyline(x, y, image_size, valid=valid)
    return image[None, :, :].astype(np.float32)


def add_white_noise_snr_preserve_quiet(
    time: np.ndarray,
    currents: np.ndarray,
    omega: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    quiet_ratio: float = 0.03,
) -> np.ndarray:
    """Add phase-wise SNR-calibrated AWGN while protecting sustained quiet intervals.

    The original Jung paper uses fault-aware limiters for intervals in which
    current is physically blocked. The five-phase IM-AA labels do not expose
    the original NPC switch topology, so V3 uses a generic short-time-RMS gate:
    noise is suppressed only where local current energy is very small. Noise
    power is then recalibrated over the full phase record to preserve the
    requested record-level SNR. This is a task adaptation, not the paper's
    exact THD+N augmentation or switch-specific limiter implementation.
    """
    t = np.asarray(time, dtype=np.float64)
    signal = np.asarray(currents, dtype=np.float64)
    speed = np.asarray(omega, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 5:
        raise ValueError(f"currents must be [T,5], got {signal.shape}")
    if t.ndim != 1 or speed.shape != t.shape or len(t) != len(signal):
        raise ValueError("time, omega and currents must have aligned lengths")
    if len(t) < 3 or not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing")
    if not np.isfinite(signal).all() or not np.isfinite(speed).all():
        raise ValueError("non-finite current or speed")
    if not -10.0 <= float(snr_db) <= 80.0:
        raise ValueError("snr_db is outside the supported range")
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    f0 = float(np.median(np.abs(speed))) / (2.0 * np.pi)
    samples_per_cycle = fs / f0 if f0 > 1e-6 else 200.0
    window = max(3, int(round(samples_per_cycle * 0.03)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    out = signal.copy()
    target_noise_ratio = 10.0 ** (-float(snr_db) / 10.0)
    for phase in range(5):
        phase_signal = signal[:, phase]
        signal_power = float(np.mean(phase_signal ** 2))
        if signal_power <= 1e-18:
            continue
        local_power = np.convolve(phase_signal ** 2, kernel, mode="same")
        local_rms = np.sqrt(np.maximum(local_power, 0.0))
        reference = float(np.percentile(local_rms, 95))
        threshold = max(quiet_ratio * reference, 1e-12)
        active = local_rms >= threshold
        if int(active.sum()) < max(8, int(0.1 * len(active))):
            active[:] = True
        noise = rng.standard_normal(len(phase_signal))
        noise[~active] = 0.0
        noise_power = float(np.mean(noise ** 2))
        if noise_power <= 1e-18:
            continue
        noise *= np.sqrt(signal_power * target_noise_ratio / noise_power)
        out[:, phase] = phase_signal + noise
    return out
