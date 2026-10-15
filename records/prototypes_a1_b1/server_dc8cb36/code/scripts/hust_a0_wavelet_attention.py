"""A0: multi-phase wavelet-attention CNN adapted from the MWA-CNN principle.

Source: Wang et al., MSSP 195 (2023) 110314. This is a HUST adaptation,
NOT their original architecture or a reproduction of their reported scores.
Fixed orthonormal Haar analysis, DWA/FAM, GN, four blocks, no LP4, no ResNet.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from v2_paifilter import RAW_POINTS, PHASES, CLASSES

WAVELET_POINTS = 512


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Critical-sampling orthonormal Haar decomposition of every channel."""
    if x.ndim != 3 or x.size(-1) % 2 != 0:
        raise ValueError("Haar DWT requires [B,C,even-length]")
    even, odd = x[..., ::2], x[..., 1::2]
    return (even + odd) / math.sqrt(2.0), (even - odd) / math.sqrt(2.0)


class SharedRMSNormalization(nn.Module):
    """Keep interphase amplitude ratios and DC polarity under common scaling.

    Unlike per-phase extrema MD, one cycle has ONE positive scale shared by
    all measured phases; an open phase is not individually amplified.
    """
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError(f"Expected [B,{PHASES},{RAW_POINTS}]")
        if not raw.is_floating_point() or not torch.isfinite(raw).all():
            raise ValueError("Raw current must be finite real floating point")
        scale = raw.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        # Block averaging of the interpolated 4096-sample cycle is anti-alias
        # smoothing, not a claim that 4096 independent samples are available.
        return F.avg_pool1d(raw / scale, kernel_size=RAW_POINTS // WAVELET_POINTS,
                            stride=RAW_POINTS // WAVELET_POINTS)


class DiscreteWaveletAttention(nn.Module):
    """Original-style DWA: [C,L] -> [2C,L/2], FAM residual enhancement.

    Signed GAP and a two-layer bottleneck reproduce the conceptual MWA-CNN
    attention; this is not noise-energy-calibrated thresholding.
    """
    def __init__(self, channels: int, dropout: float = 0.10):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        bottleneck = max(8, channels)
        self.fam = nn.Sequential(
            nn.Linear(2 * channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 2 * channels),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low, high = haar_dwt(x)
        mixed = torch.cat((low, high), dim=1)
        weights = self.fam(mixed.mean(dim=-1)).unsqueeze(-1)
        return self.dropout(mixed * (1.0 + weights))


class WaveletConvBlock(nn.Module):
    """Wavelet analysis of each channel, then convolutional phase mixing."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.10):
        super().__init__()
        self.analysis = DiscreteWaveletAttention(in_channels, dropout)
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.feature = nn.Sequential(
            nn.Conv1d(2 * in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(self.analysis(x))


class A0MultiPhaseMWACNN(nn.Module):
    """Independent wavelet-classification pipeline, NO LP4/ResNet."""
    def __init__(self, num_classes: int = CLASSES):
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two classes required")
        self.normalize = SharedRMSNormalization()
        self.layers = nn.Sequential(
            WaveletConvBlock(PHASES, 24),
            WaveletConvBlock(24, 48),
            WaveletConvBlock(48, 64),
            WaveletConvBlock(64, 96),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, raw: torch.Tensor, return_features: bool = False):
        signal = self.normalize(raw)
        encoded = self.layers(signal)
        features = self.pool(encoded).flatten(1)
        scores = self.classifier(features)
        return (scores, features) if return_features else scores


def a0_paired_loss(model: A0MultiPhaseMWACNN, clean: torch.Tensor,
                   noisy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if clean.shape != noisy.shape or labels.size(0) != clean.size(0):
        raise ValueError("Invalid paired sample batch")
    n = labels.size(0)
    scores = model(torch.cat((clean, noisy), dim=0))
    return 0.5 * (F.cross_entropy(scores[:n], labels) +
                  F.cross_entropy(scores[n:], labels))
