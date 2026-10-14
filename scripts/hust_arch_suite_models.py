"""Nine preregistered HUST architecture experiments (original V5 unaffected).

Input: raw five-phase, one full 4096-point electric cycle.  Spectral shaping
is always BEFORE the original MD (midrange centering, low-current branch,
then deterministic 4096->128 resampling).  Stage comparisons use exact same
source-only loss/selection as V5's expanded-validation protocol.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from v2_paifilter import RAW_POINTS, CNN_POINTS, PHASES, CLASSES, MDNormalization, HarmonicShaping
from v5_nahsf import NoiseAwareHarmonicFusion
from classical_baseline_models import FivePhaseResNet1D

METHODS = (
    "v5_w", "v5_wd_plain", "v5_wd_res", "v5_w_ms",
    "v5_wd_plain_ms", "v5_wd_res_ms",
    "resnet_lp3", "resnet_lp4", "resnet_nahsf",
)


class FixedLowpass4(nn.Module):
    """Phase-preserving real spectral mask k=0..4, all higher k zero."""
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        z = torch.fft.rfft(raw, n=RAW_POINTS, dim=-1, norm="ortho")
        mask = raw.new_zeros(RAW_POINTS // 2 + 1)
        mask[:5] = 1.0
        return torch.fft.irfft(z * mask[None, None, :], n=RAW_POINTS,
                                dim=-1, norm="ortho")


class MultiScaleStem(nn.Module):
    """Three aligned SAME convolutions, central crop to old valid-k16 length.

    All branches have identical sample-coordinate centers before concat.
    For 128 samples: SAME conv output length 128; [7:120] gives length 113,
    the same temporal length as valid Conv1d(kernel_size=16) in A'.
    Output channel budget is fixed at 32: (8,12,12) for kernels (7,15,31).
    """
    KERNELS = (7, 15, 31)
    CHANNELS = (8, 12, 12)
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(PHASES, c, k, stride=1, padding=k//2, bias=True)
            for k, c in zip(self.KERNELS, self.CHANNELS)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [layer(x) for layer in self.branches]
        assert all(z.size(-1) == CNN_POINTS for z in outputs)
        return torch.cat([z[..., 7:120] for z in outputs], dim=1)


class EqualParamDoubleConv(nn.Module):
    """Same parameterized path for plain and residual variants.

    In the residual case, shortcut is a parameter-free identity on 64x11.
    The comparison introduces ZERO additional trainable parameters.
    No residual skip ever bypasses the spectral front end.
    """
    def __init__(self, residual: bool):
        super().__init__()
        self.residual = bool(residual)
        self.body = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
        )
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return self.activation(x + y if self.residual else y)


class WideAEncoder(nn.Module):
    """A'-like encoder with widths 32/64/64 and unchanged 3 conv/pool stages.

    Optional +2-convolution feature block is placed AFTER the third MaxPool.
    This preserves temporal length 11, so the comparable 704->50->6 head is
    unchanged across all W / WD / MS variants.
    """
    def __init__(self, *, multiscale: bool, extra: str):
        super().__init__()
        if extra not in ("none", "plain", "residual"):
            raise ValueError(extra)
        self.multiscale = bool(multiscale)
        self.stem = nn.Sequential(
            MultiScaleStem() if multiscale else nn.Conv1d(PHASES, 32, 16),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.01), nn.MaxPool1d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, 4), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01), nn.MaxPool1d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 64, 4), nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01), nn.MaxPool1d(2),
        )
        self.extra = nn.Identity() if extra == "none" else EqualParamDoubleConv(extra == "residual")
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 11, 50), nn.LeakyReLU(0.01),
            nn.Linear(50, CLASSES),
        )

    def forward(self, signal: torch.Tensor, return_features: bool = False):
        z = self.extra(self.stage3(self.stage2(self.stem(signal))))
        if z.shape[1:] != (64, 11):
            raise RuntimeError(f"Unexpected expanded encoder output: {tuple(z.shape)}")
        logits = self.classifier(z)
        return (logits, z.flatten(1)) if return_features else logits


_SPEC = {
    "v5_w": ("wide", False, "none", "nahsf"),
    "v5_wd_plain": ("wide", False, "plain", "nahsf"),
    "v5_wd_res": ("wide", False, "residual", "nahsf"),
    "v5_w_ms": ("wide", True, "none", "nahsf"),
    "v5_wd_plain_ms": ("wide", True, "plain", "nahsf"),
    "v5_wd_res_ms": ("wide", True, "residual", "nahsf"),
    "resnet_lp3": ("resnet", False, "none", "lp3"),
    "resnet_lp4": ("resnet", False, "none", "lp4"),
    "resnet_nahsf": ("resnet", False, "none", "nahsf"),
}


class HUSTArchitectureCandidate(nn.Module):
    """Shared rigorously frozen front-end ordering across architecture arms.

    ResNet uses its EXISTING stem/layers/pool/classifier directly on already-
    normalized [B,5,128]; does NOT accidentally apply MD twice.  The original
    ResNet architecture is imported intact, with no new head or shortcut.
    """
    def __init__(self, method: str):
        super().__init__()
        if method not in _SPEC:
            raise ValueError(f"Unknown candidate {method}")
        self.method = method
        architecture, multiscale, extra, frequency = _SPEC[method]
        # Build main encoder BEFORE trainable gate, as in V5, to keep the same
        # per-seed random-number ordering across identical architecture arms.
        if architecture == "wide":
            self.backbone = WideAEncoder(multiscale=multiscale, extra=extra)
        else:
            self.backbone = FivePhaseResNet1D()
        self.normalize = MDNormalization()
        if frequency == "nahsf":
            self.shaping = NoiseAwareHarmonicFusion()
        elif frequency == "lp3":
            self.shaping = HarmonicShaping("lowpass3")
        elif frequency == "lp4":
            self.shaping = FixedLowpass4()
        else:
            raise AssertionError(frequency)

    def forward(self, raw: torch.Tensor, return_features: bool = False):
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError(f"Expected [B,{PHASES},{RAW_POINTS}], got {tuple(raw.shape)}")
        shaped = self.shaping(raw)
        signal = self.normalize(shaped)
        if self.method.startswith("resnet_"):
            # Bypass reference ResNet.forward(raw) because it expects raw 4096
            # and would apply a second (incorrect) MD normalization.
            z = self.backbone.pool(
                self.backbone.layers(self.backbone.stem(signal))
            ).flatten(1)
            logits = self.backbone.classifier(z)
            return (logits, z) if return_features else logits
        return self.backbone(signal, return_features=return_features)


def paired_ce(model: nn.Module, clean: torch.Tensor, noisy: torch.Tensor,
              labels: torch.Tensor) -> torch.Tensor:
    """Existing V5/ResNet clean-noisy jointly normalized BN training loss."""
    n = labels.size(0)
    pred = model(torch.cat((clean, noisy), dim=0))
    return 0.5 * (F.cross_entropy(pred[:n], labels) +
                  F.cross_entropy(pred[n:], labels))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def smoke() -> None:
    torch.manual_seed(42)
    raw = torch.randn(2, PHASES, RAW_POINTS)
    target = torch.tensor([1, 4])
    for name in METHODS:
        m = HUSTArchitectureCandidate(name)
        logits, features = m(raw, return_features=True)
        expected = 128 if name.startswith("resnet_") else 704
        assert logits.shape == (2, CLASSES)
        assert features.shape == (2, expected)
        loss = paired_ce(m, raw, raw + .1 * torch.randn_like(raw), target)
        loss.backward()
        assert any(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
        print(name, parameter_count(m), features.shape, flush=True)

if __name__ == "__main__":
    smoke()
