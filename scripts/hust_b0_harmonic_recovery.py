"""B0: source-supervised multi-phase complex-harmonic recovery + ResNet1D.

Research prototype; not an implementation of SSLWPT or a verified SOTA model.
Each forward takes ONLY one noisy/raw cycle, never its clean counterpart.
Clean source cycles are used solely as training targets for recovery_loss.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from v2_paifilter import RAW_POINTS, PHASES, CLASSES, MDNormalization
from classical_baseline_models import FivePhaseResNet1D

MAX_HARMONIC = 8
NOISE_BAND = (20, 80)


def _validate_cycles(x: torch.Tensor) -> None:
    if x.ndim != 3 or tuple(x.shape[1:]) != (PHASES, RAW_POINTS):
        raise ValueError(f"Expected [B,{PHASES},{RAW_POINTS}], got {tuple(x.shape)}")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Input cycles must be finite real floating point")


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """[B,P,K] complex -> [B,2P,K] real; phase order is preserved."""
    return torch.cat((x.real, x.imag), dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] != 2 * PHASES:
        raise ValueError("Complex channel shape mismatch")
    return torch.complex(x[:, :PHASES], x[:, PHASES:])


class HarmonicRecovery(nn.Module):
    """Joint phase/order residual restoration with an input-only noise proxy.

    Operates on normalized complex coefficients. Gain head initialized ~0.982
    and correction head initialized 0 => near-identity on retained harmonics.
    The original k=0 imaginary coefficient is explicitly kept zero.
    No use of known SNR, target-domain labels, or a clean reference at inference.
    """
    def __init__(self, max_harmonic: int = MAX_HARMONIC, width: int = 48):
        super().__init__()
        if not (4 <= max_harmonic <= 64) or width < 8:
            raise ValueError("Invalid recovery dimensions")
        self.max_harmonic = int(max_harmonic)
        self.input_channels = 2 * PHASES + PHASES + 1
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.gain = nn.Conv1d(width, PHASES, 1)
        self.delta = nn.Conv1d(width, 2 * PHASES, 1)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, 4.0)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    @staticmethod
    def reference_scale(spectrum: torch.Tensor) -> torch.Tensor:
        """Common five-phase fundamental scale; avoids phase-wise zero collapse."""
        return spectrum[:, :, 1].abs().square().mean(dim=1).sqrt().clamp_min(1e-4)

    @staticmethod
    def noise_features(spectrum: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Per-phase high-band energy proxy; NOT a calibrated SNR estimator."""
        power = spectrum[:, :, NOISE_BAND[0]:NOISE_BAND[1] + 1].abs().square()
        # 当前CUDA的median(dim=...)同时返回索引，在确定性模式下会直接报错。
        # 稳定排序后取下中位数，与原median的数值定义相同；固定20..80阶
        # 共61项，实际取第31项。保留原噪声代理、梯度路径及确定性设置。
        ordered = torch.sort(power, dim=-1, stable=True).values
        median_power = ordered[..., (power.size(-1) - 1) // 2]
        per_phase = median_power / scale[:, None].square().clamp_min(1e-8)
        return torch.log1p(per_phase).clamp(0.0, 30.0)

    def forward(self, raw: torch.Tensor, return_spectrum: bool = False):
        _validate_cycles(raw)
        spectrum = torch.fft.rfft(raw, dim=-1, norm="ortho")
        retained = spectrum[:, :, :self.max_harmonic + 1]
        scale = self.reference_scale(spectrum)
        relative = retained / scale[:, None, None]
        noise = self.noise_features(spectrum, scale)
        features = torch.cat((
            complex_to_channels(relative),
            noise.unsqueeze(-1).expand(-1, -1, retained.size(-1)),
            torch.linspace(0, 1, retained.size(-1), device=raw.device,
                           dtype=raw.dtype).view(1, 1, -1).expand(raw.size(0), -1, -1),
        ), dim=1)
        hidden = self.encoder(features)
        gain = torch.sigmoid(self.gain(hidden))
        # Bounded per-phase complex correction; full-scale inversion is not assumed.
        delta = 0.35 * torch.tanh(self.delta(hidden))
        estimate = gain * relative + channels_to_complex(delta)
        estimated_complex = estimate * scale[:, None, None]
        # Real-valued waveform requires a real DC coefficient.
        estimated_complex = torch.cat(
            (estimated_complex[:, :, :1].real.to(spectrum.dtype),
             estimated_complex[:, :, 1:]), dim=-1
        )
        padded = torch.cat((
            estimated_complex,
            spectrum.new_zeros(raw.size(0), PHASES,
                               spectrum.size(-1) - estimated_complex.size(-1)),
        ), dim=-1)
        restored = torch.fft.irfft(padded, n=RAW_POINTS, dim=-1, norm="ortho")
        return (restored, estimated_complex) if return_spectrum else restored


def harmonic_recovery_loss(estimate: torch.Tensor, clean_raw: torch.Tensor) -> torch.Tensor:
    """Normalized source-only complex recovery target. No target-domain inputs."""
    _validate_cycles(clean_raw)
    if not estimate.is_complex() or estimate.shape[:2] != clean_raw.shape[:2]:
        raise ValueError("Recovery estimate has wrong dimensions")
    reference_fft = torch.fft.rfft(clean_raw, dim=-1, norm="ortho")
    target = reference_fft[:, :, :estimate.size(-1)]
    scale = HarmonicRecovery.reference_scale(reference_fft).detach()
    difference = (estimate - target) / scale[:, None, None]
    # Huber on real/imag so the dominant fundamental cannot dominate by power.
    return F.smooth_l1_loss(difference.real, torch.zeros_like(difference.real)) + \
           F.smooth_l1_loss(difference.imag, torch.zeros_like(difference.imag))


class B0HarmonicRecoveryNet(nn.Module):
    """Restoration is learned; waveforms reconstructed before existing ResNet."""
    def __init__(self, max_harmonic: int = MAX_HARMONIC):
        super().__init__()
        self.restorer = HarmonicRecovery(max_harmonic)
        self.normalize = MDNormalization()
        self.backbone = FivePhaseResNet1D()

    def forward(self, raw: torch.Tensor, return_recovery: bool = False):
        restored, spectrum = self.restorer(raw, return_spectrum=True)
        signal = self.normalize(restored)
        # The reused backbone.forward() applies MD itself; bypass it to avoid
        # an accidental second normalization.
        feature = self.backbone.pool(
            self.backbone.layers(self.backbone.stem(signal))
        ).flatten(1)
        logits = self.backbone.classifier(feature)
        return (logits, spectrum) if return_recovery else logits


def b0_paired_loss(model: B0HarmonicRecoveryNet, clean: torch.Tensor,
                   noisy: torch.Tensor, labels: torch.Tensor,
                   lambda_rec: float = 0.2):
    if lambda_rec < 0 or clean.shape != noisy.shape or labels.shape[0] != clean.shape[0]:
        raise ValueError("Invalid paired training batch / loss weight")
    n = labels.size(0)
    logits, recovered = model(torch.cat([clean, noisy], dim=0), return_recovery=True)
    ce = 0.5 * (F.cross_entropy(logits[:n], labels) +
                F.cross_entropy(logits[n:], labels))
    rec = harmonic_recovery_loss(recovered[n:], clean)
    return ce + lambda_rec * rec, {"ce": ce.detach(), "rec": rec.detach()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
