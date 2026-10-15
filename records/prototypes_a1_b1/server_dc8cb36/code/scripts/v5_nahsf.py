"""HUST V5: Noise-Aware Harmonic Selective Fusion (NA-HSF).

A' remains the frozen 25,620-parameter V2 encoder/classifier.  Only the
pre-normalization spectral block changes.  The LP3/LP4 convex combination is
mathematically identical to a sample-dependent gain on harmonic order 4;
the novelty hypothesis concerns the *noise-aware gate*, not two networks.

No clean reference, fault label, target SNR, target statistics, or target-domain
learning is used by the gate.  Target-Dev is evaluation only.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from v2_paifilter import (
    RAW_POINTS, CNN_POINTS, PHASES, CLASSES, MDNormalization, OriginalCNN,
    paired_cross_entropy,
)

NOISE_START = 12
NOISE_END = 30
GATE_FEATURES = 3
GATE_HIDDEN = 8
LOG_RATIO_CLIP = 6.0


class NoiseAwareHarmonicFusion(nn.Module):
    """Per-sample, five-phase-shared gate between LP3 and LP4.

    Input: raw, unnormalized complete electrical cycles [B, 5, 4096].
    Output: same shape.  The three descriptors are all from current input:

      1. low_fraction = sum(P[1:4]) / sum(P[1:31])
      2. log_ratio4 = clamp(log10(P4 / median(P[12:31])), -6, 6)/6
      3. noise_fraction = sum(P[12:31]) / sum(P[1:31])

    P[k] is five-phase mean squared magnitude of the orthonormal rFFT.
    The median is the 10th order statistic of 19 entries (stable CUDA sort).
    The scaling avoids huge dynamic range / non-finite gradients when the
    noise floor is nearly zero.  This descriptor is a *proxy*, not true SNR.

    alpha = sigmoid(Linear(8,1)(ReLU(Linear(3,8)(descriptors)))).
    Its final layer is zero-initialized so alpha == 0.5 on every input at
    initialization.  No gradient detachment at the learned gain.  k=0..3
    passed, k=4 multiplied by alpha, and k>=5 discarded.  The gain is real
    and five-phase shared so it does not rotate cross-phase FFT coefficients.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(GATE_FEATURES, GATE_HIDDEN),
            nn.ReLU(),
            nn.Linear(GATE_HIDDEN, 1),
        )
        # Architecture is initialized only after CNN for V4 seed parity.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def descriptors(spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim != 3 or spectrum.shape[1:] != (PHASES, RAW_POINTS // 2 + 1):
            raise ValueError(f"Expected complex spectrum [B,{PHASES},{RAW_POINTS//2+1}]")
        with torch.no_grad():
            p = spectrum.abs().square().mean(dim=1)  # [B, 2049]
            p_low = p[:, 1:4].sum(dim=-1)
            p_total = p[:, 1:31].sum(dim=-1)
            p_hi = p[:, NOISE_START:NOISE_END + 1].sum(dim=-1)
            band = p[:, NOISE_START:NOISE_END + 1]
            # GPU torch.median(dim) with indices is not deterministic on all
            # configurations. Sorting 19 entries gives identical median values.
            floor = band.sort(dim=-1, stable=True).values[:, 9]
            eps = torch.finfo(p.dtype).eps
            low_fraction = p_low / p_total.clamp_min(eps)
            noise_fraction = p_hi / p_total.clamp_min(eps)
            ratio = p[:, 4].clamp_min(eps) / floor.clamp_min(eps)
            log_ratio = torch.log10(ratio).clamp(-LOG_RATIO_CLIP, LOG_RATIO_CLIP) / LOG_RATIO_CLIP
            features = torch.stack((low_fraction, log_ratio, noise_fraction), dim=-1)
            if not bool(torch.isfinite(features).all()):
                raise FloatingPointError("Nonfinite NA-HSF input descriptors")
            return features

    def forward(self, raw: torch.Tensor, return_gate: bool = False,
                return_descriptors: bool = False):
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError(f"Expected raw [B,{PHASES},{RAW_POINTS}], got {tuple(raw.shape)}")
        if not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
            raise ValueError("Raw cycles must be finite floating-point arrays")
        spectrum = torch.fft.rfft(raw, n=RAW_POINTS, dim=-1, norm="ortho")
        z = self.descriptors(spectrum)
        alpha = torch.sigmoid(self.gate(z)).squeeze(-1)
        # Preserve complex phase coefficients exactly; no learned phase shift.
        mask = raw.new_zeros((raw.shape[0], RAW_POINTS // 2 + 1))
        mask[:, :4] = 1.0
        mask[:, 4] = alpha
        restored = torch.fft.irfft(spectrum * mask.unsqueeze(1), n=RAW_POINTS,
                                    dim=-1, norm="ortho")
        if return_descriptors:
            return restored, alpha, z
        if return_gate:
            return restored, alpha
        return restored


class NAHSFCNN(nn.Module):
    """NA-HSF + frozen A' topology; C is the only trainable addition."""

    def __init__(self) -> None:
        super().__init__()
        # Critical RNG ordering: V4 shaping uses no RNG before OriginalCNN.
        # Construct backbone first, then initialize MLP, preserving the V4
        # initial CNN bytes for each fixed torch seed.
        self.backbone = OriginalCNN()
        self.shaping = NoiseAwareHarmonicFusion()
        self.normalize = MDNormalization()

    def forward(self, raw: torch.Tensor, return_gate: bool = False,
                return_descriptors: bool = False):
        if return_descriptors:
            shaped, alpha, z = self.shaping(raw, return_descriptors=True)
        elif return_gate:
            shaped, alpha = self.shaping(raw, return_gate=True)
        else:
            shaped = self.shaping(raw)
        logits = self.backbone(self.normalize(shaped))
        if return_descriptors:
            return logits, alpha, z
        if return_gate:
            return logits, alpha
        return logits


def parameter_counts(model: NAHSFCNN) -> dict[str, int]:
    cnn = sum(p.numel() for p in model.backbone.parameters())
    gate = sum(p.numel() for p in model.shaping.parameters())
    return {"cnn": cnn, "spectral": gate, "total": cnn + gate}


def smoke() -> None:
    torch.manual_seed(42)
    model = NAHSFCNN()
    raw = torch.randn(3, PHASES, RAW_POINTS)
    logits, alpha, z = model(raw, return_descriptors=True)
    assert logits.shape == (3, CLASSES)
    assert z.shape == (3, GATE_FEATURES)
    assert torch.equal(alpha, torch.full_like(alpha, .5))
    loss = paired_cross_entropy(model, raw, raw + .1 * torch.randn_like(raw),
                                torch.tensor([0, 1, 2]))
    loss.backward()
    grad = model.shaping.gate[-1].bias.grad
    assert grad is not None and bool(torch.isfinite(grad).all())
    print("V5 NA-HSF SMOKE PASS", parameter_counts(model), "loss=", round(float(loss), 6))


if __name__ == "__main__":
    smoke()
