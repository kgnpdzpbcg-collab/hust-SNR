"""A1: multiphase noise-sensitive wavelet CNN (standalone alternative to LP4).

Changes relative to archived A0 (left unmodified):
  - robust shared per-cycle k=1 amplitude scale (five-phase median);
  - Haar low-frequency path is never gated;
  - high-frequency wavelet details are only attenuated, never amplified;
  - RMS-based low/detail statistics replace signed GAP;
  - learned correction to a bounded noise-ratio shrinkage prior.
Inspired by MWA-CNN's DWT+attention architecture; not a reproduction of it.
No measured-domain data, labels, clean reference, or known SNR at inference.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from v2_paifilter import RAW_POINTS, PHASES, CLASSES
from hust_a0_wavelet_attention import haar_dwt, WAVELET_POINTS


class RobustPhaseScale(nn.Module):
    """Normalize by median k=1 amplitude across the five synchronized phases.

    This statistic ignores overall broadband RMS inflation at low SNR and
    retains sign, phase identity, relative phase amplitudes and missing phases.
    It is an input-dependent estimate; NOT ground-truth current amplitude.
    """
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError(f"Expected [B,{PHASES},{RAW_POINTS}]")
        if not raw.is_floating_point() or not torch.isfinite(raw).all().item():
            raise ValueError("Input must contain finite float current samples")
        fundamental = torch.fft.rfft(raw, dim=-1, norm="ortho")[:, :, 1].abs()
        # Stable sort avoids nondeterministic CUDA median-with-indices backward.
        middle = torch.sort(fundamental, dim=1, stable=True).values[:, PHASES // 2]
        scale = (2.0 / math.sqrt(RAW_POINTS)) * middle
        normalized = raw / scale.clamp_min(1e-3).view(-1, 1, 1)
        return F.avg_pool1d(normalized, kernel_size=RAW_POINTS // WAVELET_POINTS,
                            stride=RAW_POINTS // WAVELET_POINTS)


class ReliabilityDWA(nn.Module):
    """Orthonormal Haar analysis followed by a reliability-conditioned detail gate.

    Gate = sigmoid(log(low_RMS/high_RMS) + learnable_correction).
    With a zero-initialized last layer this is low_RMS/(low_RMS+high_RMS)
    up to epsilon, and always lies in (0,1). The low band is exactly
    unchanged by this layer. Statistics are input-derived and nonnegative.
    """
    def __init__(self, channels: int, dropout: float = 0.10,
                 epsilon: float = 1e-4):
        super().__init__()
        if channels < 1 or not (0 <= dropout < 1) or epsilon <= 0:
            raise ValueError("Invalid reliability gate configuration")
        self.epsilon = epsilon
        hidden = max(12, channels)
        self.adjustment = nn.Sequential(
            nn.Linear(3 * channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(self.adjustment[-1].weight)
        nn.init.zeros_(self.adjustment[-1].bias)
        self.detail_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        low, high = haar_dwt(x)
        low_rms = low.square().mean(-1).add(1e-8).sqrt()
        high_rms = high.square().mean(-1).add(1e-8).sqrt()
        # Actual energy is visible even when the wavelet detail sums to zero.
        statistics = torch.cat((
            torch.log1p(low_rms), torch.log1p(high_rms),
            torch.log1p(high_rms / (low_rms + self.epsilon)),
        ), dim=1)
        prior_logit = torch.log(low_rms + self.epsilon) - \
                      torch.log(high_rms + self.epsilon)
        adjustment = self.adjustment(statistics)
        gate = torch.sigmoid(prior_logit + adjustment)
        low_safe = low  # no lowband gating and no lowband dropout
        detail = self.detail_dropout(high * gate.unsqueeze(-1))
        out = torch.cat((low_safe, detail), dim=1)
        return (out, gate) if return_gate else out


class ReliabilityWaveletBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.wavelet = ReliabilityDWA(in_channels)
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.mix = nn.Sequential(
            nn.Conv1d(2 * in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.wavelet(x))


class A1NoiseAwareWaveletCNN(nn.Module):
    """Four-stage [B,5,4096] -> [B,6] wavelet-only classifier."""
    def __init__(self, num_classes: int = CLASSES):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Need >=2 fault classes")
        self.normalize = RobustPhaseScale()
        self.blocks = nn.Sequential(
            ReliabilityWaveletBlock(PHASES, 24),
            ReliabilityWaveletBlock(24, 48),
            ReliabilityWaveletBlock(48, 64),
            ReliabilityWaveletBlock(64, 96),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, raw: torch.Tensor, return_features: bool = False):
        x = self.blocks(self.normalize(raw))
        features = self.pool(x).flatten(1)
        output = self.classifier(features)
        return (output, features) if return_features else output


def a1_paired_loss(model: A1NoiseAwareWaveletCNN,
                   clean: torch.Tensor, noisy: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
    if clean.shape != noisy.shape or labels.ndim != 1 or labels.size(0) != clean.size(0):
        raise ValueError("Paired input/label shapes mismatch")
    n = labels.size(0)
    logits = model(torch.cat((clean, noisy), dim=0))
    return 0.5 * (F.cross_entropy(logits[:n], labels) +
                  F.cross_entropy(logits[n:], labels))
