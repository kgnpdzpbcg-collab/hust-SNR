"""B1: source-supervised, phase-structure-preserving harmonic denoising.

Independent B0 successor, NOT an incremental modification of historical B0
weights. Uses k=0..4 because fixed-weight B0 intervention found k5..8
degrades strong-noise performance; this observation is *diagnostic*,
not a target-label hyperparameter selection protocol for B1.
The learned part operates on noisy observation ONLY at inference:
  1) median high-band noise proxy (same measured cycle);
  2) positive, frequency-calibrated estimated noise variance;
  3) phase-invariant five-phase reliability statistics;
  4) phase-preserving bounded gain (no arbitrary complex additive term);
  5) fused original MD shape + common-scale weak-phase evidence.
At training: noisy->clean source complex recovery + phase-strength + noise
calibration, followed by CE of clean/noisy source views.
No VSD physics or correct-noise-spectrum generalization is claimed.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from v2_paifilter import (
    RAW_POINTS, PHASES, CLASSES, CNN_POINTS, MDNormalization,
    deterministic_linear_resample,
)
from classical_baseline_models import FivePhaseResNet1D
from hust_b0_harmonic_recovery import _validate_cycles

MAX_ORDER = 4
NOISE_BAND = (20, 80)


def _middle_five(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Median-of-five values, CUDA deterministic (stable sort)."""
    if x.size(dim) != PHASES:
        raise ValueError("Expected one scalar per measured phase")
    return torch.sort(x, dim=dim, stable=True).values.select(dim, PHASES // 2)


def shared_harmonic_scale(spectrum: torch.Tensor) -> torch.Tensor:
    """No ground truth or label: common k=1 phase median, orthonormal FFT."""
    return _middle_five(spectrum[:, :, 1].abs(), 1).clamp_min(1e-4)


def phase_strength(spectrum: torch.Tensor) -> torch.Tensor:
    """Relative harmonic-energy envelope (zero phase -> zero strength).

    Input complex [B,5,>=5]. The ratio is smooth almost everywhere and
    invariant to global time shifts (harmonic phase rotations).
    """
    if not spectrum.is_complex() or spectrum.ndim != 3 or spectrum.size(1) != PHASES \
            or spectrum.size(2) < 5:
        raise ValueError("Need [B,5,>=5] complex harmonics")
    energy = spectrum[:, :, 1:5].abs().square().sum(dim=-1).add(1e-12).sqrt()
    median = _middle_five(energy, 1).clamp_min(1e-4)
    return energy / median[:, None]


class PhaseAwareHarmonicRestorer(nn.Module):
    """Observed-phase-preserving noise-conditioned shrinkage for orders k0..k4.

    No direct additive complex correction: output_k = gain_k * observed_k,
    0<gain<1. Gains depend on gauge-invariant multi-phase Gram statistics,
    signal/noise amplitude, harmonic order and learned noise calibration.
    The k=0 imaginary part remains zero by construction.
    """
    def __init__(self, width: int = 48):
        super().__init__()
        if width < 12:
            raise ValueError("Width too small")
        self.log_noise_order_scale = nn.Parameter(torch.zeros(MAX_ORDER + 1))
        self.adjustment = nn.Sequential(
            nn.Conv1d(5 * PHASES + 1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, PHASES, 1),
        )
        nn.init.zeros_(self.adjustment[-1].weight)
        nn.init.zeros_(self.adjustment[-1].bias)

    @staticmethod
    def noise_proxy(full_spectrum: torch.Tensor,
                    shared_scale: torch.Tensor) -> torch.Tensor:
        """Per-phase relative power; spectral shape may shift across domains."""
        values = full_spectrum[:, :, NOISE_BAND[0]:NOISE_BAND[1] + 1].abs().square()
        sorted_power = torch.sort(values, dim=-1, stable=True).values
        median_power = sorted_power[..., (values.size(-1) - 1) // 2]
        return median_power / shared_scale[:, None].square().clamp_min(1e-8)

    def forward(self, raw: torch.Tensor, return_details: bool = False):
        _validate_cycles(raw)
        full = torch.fft.rfft(raw, dim=-1, norm="ortho")
        scale = shared_harmonic_scale(full)
        normalized = full[:, :, :MAX_ORDER + 1] / scale[:, None, None]
        signal_power = normalized.abs().square()
        proxy = self.noise_proxy(full, scale)
        # Extrapolatable variance model: noise power scales linearly with
        # observed high-frequency power; learn positive k-dependent factor.
        calibrated = proxy[:, :, None] * \
                     torch.exp(self.log_noise_order_scale).view(1, 1, -1)
        variance = calibrated.clamp_min(1e-10)

        # Gram-derived cos coherence at the *same harmonic order*, invariant
        # under a common cycle rotation. This informs gains, never injects
        # a potentially false current from another phase.
        gram = normalized[:, :, None, :] * normalized[:, None, :, :].conj()
        amplitudes = normalized.abs()
        denominator = (amplitudes[:, :, None, :] *
                       amplitudes[:, None, :, :]).clamp_min(1e-7)
        phase_consistency = (gram.real / denominator).mean(dim=2)
        phase_consistency = phase_consistency.clamp(-1.0, 1.0)
        relative_power = signal_power / \
                         _middle_five(signal_power, dim=1).clamp_min(1e-7)[:, None, :]
        log_snr = (torch.log(signal_power + 1e-8) -
                   torch.log(variance + 1e-8)).clamp(-25, 25)

        features = torch.cat((
            torch.log1p(signal_power),
            torch.log1p(relative_power),
            phase_consistency,
            torch.log1p(proxy).unsqueeze(-1).expand_as(signal_power),
            torch.tanh(log_snr / 8.0),
            torch.linspace(0, 1, MAX_ORDER + 1, device=raw.device,
                           dtype=raw.dtype).view(1, 1, -1).expand(raw.size(0), -1, -1),
        ), dim=1)
        correction = 2.0 * torch.tanh(self.adjustment(features))
        gain = torch.sigmoid(log_snr + correction)
        spectrum_hat = full[:, :, :MAX_ORDER + 1] * gain
        pad = full.new_zeros(raw.size(0), PHASES,
                             full.size(-1) - (MAX_ORDER + 1))
        restored = torch.fft.irfft(torch.cat((spectrum_hat, pad), -1),
                                   n=RAW_POINTS, dim=-1, norm="ortho")
        if return_details:
            return restored, spectrum_hat, variance, gain
        return restored


class B1PhaseAwareRecoveryNet(nn.Module):
    """Two physical views: per-phase MD shape and common-phase amplitude."""
    def __init__(self):
        super().__init__()
        self.restorer = PhaseAwareHarmonicRestorer()
        self.normalize = MDNormalization()
        self.backbone = FivePhaseResNet1D()
        # These are parallel *features*, not two stacked MD operations.
        self.backbone.stem[0] = nn.Conv1d(2 * PHASES, 32, 7, padding=3, bias=False)

    @staticmethod
    def _physical_view(raw: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(raw, dim=-1, norm="ortho")
        # k=1: median amplitude across phases excludes one/two open phases
        reference = shared_harmonic_scale(spectrum)
        # k1 magnitude of A*cos has orthonormal amplitude A*sqrt(N)/2.
        amp = (2.0 / math.sqrt(RAW_POINTS)) * reference
        normalized = raw / amp.clamp_min(1e-3).view(-1, 1, 1)
        return deterministic_linear_resample(normalized, CNN_POINTS)

    def forward(self, raw: torch.Tensor, return_recovery: bool = False):
        restored, harmonics, noise_variance, gain = self.restorer(
            raw, return_details=True)
        md_shape = self.normalize(restored)
        relative_phase = self._physical_view(restored)
        signal = torch.cat((md_shape, relative_phase), dim=1)
        hidden = self.backbone.pool(
            self.backbone.layers(self.backbone.stem(signal))
        ).flatten(1)
        logits = self.backbone.classifier(hidden)
        if return_recovery:
            return logits, harmonics, noise_variance, gain
        return logits


def b1_recovery_objective(noisy: torch.Tensor, clean: torch.Tensor,
                          recovered: torch.Tensor, noise_variance: torch.Tensor,
                          *, w_complex: float = 1.0, w_phase: float = 0.4,
                          w_noise: float = 0.1):
    """Only uses CLEAN SOURCE as the training target, never target measured data."""
    _validate_cycles(noisy)
    _validate_cycles(clean)
    if noisy.shape != clean.shape or recovered.shape != (noisy.size(0), PHASES, MAX_ORDER + 1) \
            or noise_variance.shape != recovered.shape or not recovered.is_complex():
        raise ValueError("Recovery supervision shape/type mismatch")
    if min(w_complex, w_phase, w_noise) < 0:
        raise ValueError("Negative loss coefficient")
    clean_fft = torch.fft.rfft(clean, dim=-1, norm="ortho")
    noisy_fft = torch.fft.rfft(noisy, dim=-1, norm="ortho")
    scale = shared_harmonic_scale(noisy_fft).detach()
    target = clean_fft[:, :, :MAX_ORDER + 1]
    residual = (recovered - target) / scale[:, None, None]
    # Each complex coordinate is normalized by the shared phase amplitude.
    complex_term = (
        F.smooth_l1_loss(residual.real, torch.zeros_like(residual.real)) +
        F.smooth_l1_loss(residual.imag, torch.zeros_like(residual.imag))
    )
    # Explicitly preserve weak/open-phase k1..4 energy relative to other phases,
    # rather than imposing the healthy-only symmetry of an intact inverter.
    phys = F.smooth_l1_loss(torch.log1p(phase_strength(recovered)),
                            torch.log1p(phase_strength(target)))
    known_noise = ((noisy_fft - clean_fft)[:, :, :MAX_ORDER + 1] /
                   scale[:, None, None]).abs().square()
    # Single complex coefficient has highly variable observed power.
    # Compare phase-and-batch means, not per-sample point estimates.
    measured_mean = known_noise.mean(dim=(0, 1))
    predicted_mean = noise_variance.mean(dim=(0, 1))
    noise_cal = F.smooth_l1_loss(
        torch.log(predicted_mean + 1e-9),
        torch.log(measured_mean + 1e-9))
    total = w_complex * complex_term + w_phase * phys + w_noise * noise_cal
    return total, {"complex": complex_term.detach(),
                   "phase": phys.detach(), "noise": noise_cal.detach()}


def b1_pretrain_loss(model: B1PhaseAwareRecoveryNet,
                     clean: torch.Tensor, noisy: torch.Tensor):
    _, spec, variance, _ = model.restorer(noisy, return_details=True)
    return b1_recovery_objective(noisy, clean, spec, variance)


def b1_paired_loss(model: B1PhaseAwareRecoveryNet,
                   clean: torch.Tensor, noisy: torch.Tensor,
                   labels: torch.Tensor, lambda_restore: float = 1.0):
    if lambda_restore < 0 or labels.ndim != 1 or labels.size(0) != clean.size(0) \
            or clean.shape != noisy.shape:
        raise ValueError("Paired loss input mismatch")
    n = labels.size(0)
    logits, recovered, variance, _ = model(
        torch.cat((clean, noisy), dim=0), return_recovery=True)
    ce = 0.5 * (F.cross_entropy(logits[:n], labels) +
                F.cross_entropy(logits[n:], labels))
    restoration, parts = b1_recovery_objective(
        noisy, clean, recovered[n:], variance[n:])
    total = ce + lambda_restore * restoration
    return total, {"ce": ce.detach(), "restore": restoration.detach(), **parts}
