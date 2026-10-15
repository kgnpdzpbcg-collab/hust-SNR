"""Source-only classical baselines for the HUST five-phase open-circuit task.

Feature baselines use the same raw 4096-point electrical cycle as V5; the
standardization model is fit on Source-Train only. ResNet uses exactly the
existing differentiable 4096->128 MD preprocessing with NO harmonic filter.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from v2_paifilter import MDNormalization, RAW_POINTS, PHASES, CLASSES

FEATURE_VERSION = "five_phase_raw_md_stat_harmonic_v1"


def md_normalize_numpy(raw: np.ndarray) -> np.ndarray:
    """Same per-phase MD rule as V5, BEFORE any optional 128-point resampling.

    [N,5,4096]; low-current phases (<.2 average centered amplitude) divide by
    100, others by their centered maximum absolute value. No data-set fit.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (PHASES, RAW_POINTS):
        raise ValueError(f"Expected [N,{PHASES},{RAW_POINTS}], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite input current")
    center = (raw.max(axis=-1, keepdims=True) + raw.min(axis=-1, keepdims=True)) / 2
    x = raw - center
    small = np.mean(np.abs(x), axis=-1, keepdims=True) < .2
    amp = np.max(np.abs(x), axis=-1, keepdims=True)
    divisor = np.where(small, 100., np.maximum(amp, 1e-8))
    return x / divisor


def engineered_features(raw: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Physically motivated, label-free features, 5 phases × (8 time + 8
    harmonic amplitude + 8 harmonic real/imag) = 120 features.

    Harmonic amplitude and phase use frequency *order* (integer k per full
    electrical cycle), not absolute Hz. The FFT remains phase-preserving;
    complex k1..4 coefficients can capture phase imbalance. No fault label,
    rotational speed metadata, or target-domain statistics are used.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (PHASES, RAW_POINTS):
        raise ValueError("engineered_features requires [N,5,4096]")
    blocks = []
    for i in range(0, len(raw), batch_size):
        x = md_normalize_numpy(raw[i:i + batch_size]).astype(np.float64)
        mu = x.mean(axis=-1)
        delta = x-mu[...,None]
        sd = np.sqrt(np.mean(delta**2,axis=-1)).clip(min=1e-8)
        meanabs = np.mean(np.abs(x),axis=-1)
        rms = np.sqrt(np.mean(x**2,axis=-1))
        peak = np.max(np.abs(x),axis=-1)
        tim = np.stack((mu,rms,meanabs,sd,
                        x.max(-1)-x.min(-1),
                        peak/(rms+1e-8),
                        np.mean((delta/sd[...,None])**3,axis=-1),
                        np.mean((delta/sd[...,None])**4,axis=-1)),axis=-1)
        # rFFT(N) / N yields an amplitude-scale invariant to cycle length.
        ft = np.fft.rfft(x,axis=-1) / RAW_POINTS
        magnitudes = np.abs(ft[...,1:9])  # [B,5,8]
        low_complex = np.stack((ft[...,1:5].real,ft[...,1:5].imag),axis=-1).reshape(len(x),PHASES,8)
        features = np.concatenate((tim,magnitudes,low_complex),axis=-1).reshape(len(x), -1)
        if features.shape[1] != 120 or not np.isfinite(features).all():
            raise FloatingPointError("Classical feature integrity failure")
        blocks.append(features.astype(np.float32))
    return np.concatenate(blocks,axis=0) if blocks else np.empty((0,120),dtype=np.float32)


class ResidualBlock(nn.Module):
    """Conventional two-layer 1D basic residual block; no frequency prior."""
    def __init__(self, in_channels: int, channels: int, stride: int=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels,channels,3,stride=stride,padding=1,bias=False),
            nn.BatchNorm1d(channels),nn.ReLU(inplace=True),
            nn.Conv1d(channels,channels,3,padding=1,bias=False),
            nn.BatchNorm1d(channels))
        self.shortcut = (nn.Identity() if in_channels==channels and stride==1
                         else nn.Sequential(nn.Conv1d(in_channels,channels,1,stride=stride,bias=False),
                                            nn.BatchNorm1d(channels)))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class FivePhaseResNet1D(nn.Module):
    """Raw [B,5,4096] -> V5 MD [B,5,128] -> 3 residual stages -> logits.

    Stem 5->32, stages (32,64,128) ×2 blocks, global average pooling and
    linear 128->6.  All parameters initialized from each frozen seed.
    """
    def __init__(self):
        super().__init__()
        self.normalize = MDNormalization()
        self.stem = nn.Sequential(nn.Conv1d(PHASES,32,7,padding=3,bias=False),
                                  nn.BatchNorm1d(32),nn.ReLU(inplace=True))
        self.layers = nn.Sequential(
            ResidualBlock(32,32),ResidualBlock(32,32),
            ResidualBlock(32,64,stride=2),ResidualBlock(64,64),
            ResidualBlock(64,128,stride=2),ResidualBlock(128,128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128,CLASSES)

    def forward(self, raw: torch.Tensor, return_features: bool=False):
        if raw.ndim!=3 or tuple(raw.shape[1:])!=(PHASES,RAW_POINTS):
            raise ValueError("ResNet requires [B,5,4096]")
        feature = self.pool(self.layers(self.stem(self.normalize(raw)))).flatten(1)
        logits = self.classifier(feature)
        return (logits,feature) if return_features else logits


def smoke():
    rng=np.random.default_rng(0)
    raw=rng.standard_normal((4,PHASES,RAW_POINTS)).astype(np.float32)
    f=engineered_features(raw)
    assert f.shape==(4,120) and np.isfinite(f).all()
    model=FivePhaseResNet1D()
    logits,emb=model(torch.from_numpy(raw),True)
    assert logits.shape==(4,CLASSES) and emb.shape==(4,128)
    logits.square().mean().backward()
    assert model.stem[0].weight.grad is not None
    print("Classical feature/ResNet smoke passed",f.shape,
          "ResNet parameters",sum(p.numel() for p in model.parameters()))

if __name__=="__main__":smoke()
