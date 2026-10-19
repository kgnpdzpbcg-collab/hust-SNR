"""Paper-inspired comparison architectures for the HUST five-phase IM-AA task.

JungModifiedCNN: dilated Conv1D + residual block + GAP.
HangWCNN: two five-phase VSD trajectory channels + fixed Haar LL pooling + 2D CNN.

These are transparent approximate reproductions/adaptations, not author-released code.
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
    """Hang-inspired 2-D WCNN for two VSD trajectory-image channels.

    This reconstruction chooses Conv channels 4, 8, 16, 32, 64 because the paper
    does not provide a complete layer-by-layer parameter table. Fixed Haar LL
    pooling follows conv1 and conv2; a flattened linear classifier follows conv5.
    Input is [B,2,64,64] by default. The 5-phase image adapter is in this module.
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 6,
        image_size: int = 64,
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

    Channel 0 is the alpha1-beta1 plane; channel 1 is alpha3-beta3. Per-phase
    per-phase DC offsets are removed by mean-centering, but the four VSD components share one amplitude
    scale so the weak harmonic-plane magnitude is not independently amplified.
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
