"""HUST-SNR V4: isolated fourth-harmonic recovery on a strong 3-harmonic prior.

The network is the frozen V2 25,620-parameter 1D-CNN (A'), NOT a CAE:
no decoder, reconstruction loss, DML, or target-domain adaptation.
The input pipeline uses V2's 4096-point cycle, spectral shaping before
MD normalization, and 128-point CNN input. The lowpass3 control intentionally
calls V2's exact shaping module for audit parity.

The HPASF gate is intentionally MINIMAL and a hypothesis, not an established
noise estimator or a guaranteed improvement over lowpass3 / lowpass4.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from v2_paifilter import (RAW_POINTS, PHASES, CNN_POINTS, CLASSES,
                          HarmonicShaping, MDNormalization, OriginalCNN)

METHODS = ("lowpass3", "lowpass4", "static4", "hpasf")
NOISE_START = 12              # inclusive, harmonic ORDER (NOT Hz)
NOISE_END = 30                # inclusive
INITIAL_GAIN4 = 0.05          # almost lowpass3, but sigmoid has usable gradient


class FourthHarmonicShaping(nn.Module):
    """Pass DC + harmonics 1:3 and optionally part/all of harmonic 4.

    All five phases share the same *real-valued* fourth-harmonic gain;
    therefore the gate does not explicitly rotate phase or mix channels.
    Frequencies >=5 are suppressed in all three modes.

    - lowpass4: deterministic g4=1.
    - static4: scalar learned g4=sigmoid(a), fixed across all inputs.
    - hpasf: per-sample g4=sigmoid(a)*r4 where
        r4=clamp(1-median_k(phase_mean(|X_k|^2))/phase_mean(|X_4|^2),0,1)
        for k=12,...,30. The estimated floor never uses Clean reference,
        a fault label, SNR tags, or target statistics, even at inference.

    IMPORTANT: the high-frequency median is ONLY a proxy for noise power;
    clean harmonics may inflate it, especially at high SNR.
    """
    def __init__(self, mode: str, raw_points: int = RAW_POINTS,
                 initial_gain4: float = INITIAL_GAIN4):
        super().__init__()
        if mode not in ("lowpass4", "static4", "hpasf"):
            raise ValueError(f"Invalid fourth-harmonic mode {mode!r}")
        if raw_points != RAW_POINTS:
            raise ValueError("Frozen V4 raw cycle length must be 4096")
        if not 0 < initial_gain4 < 1:
            raise ValueError("initial_gain4 must lie strictly between 0 and 1")
        self.mode = mode
        self.raw_points = raw_points
        self.register_parameter("gate_logit", None)
        if mode in ("static4", "hpasf"):
            # No RNG consumed, so all modes get paired identical CNN initializations.
            self.gate_logit = nn.Parameter(torch.tensor(
                math.log(initial_gain4 / (1.0 - initial_gain4)),
                dtype=torch.float32))

    @staticmethod
    def noise_reliability(spectrum: torch.Tensor) -> torch.Tensor:
        """Return per-window r4 [B] from the *current* raw FFT only.

        Avoid gradient paths through the non-smooth median. No trainable
        preprocessing precedes the FFT in this frozen experiment, so the
        scalar learned fourth-harmonic gate still receives correct gradients.
        """
        with torch.no_grad():
            p = spectrum.abs().square().mean(dim=1)  # [B, 2049], phases pooled
            p4 = p[:, 4]
            # CUDA median(dim=...) 同时产生索引，不能在严格确定性模式运行。
            # 本协议固定取12–30阶共19个功率值；稳定排序后第10个值就是
            # 原中位数。仅替换取值算子，不改变噪声底公式或梯度隔离规则。
            band = p[:, NOISE_START:NOISE_END + 1]
            floor = band.sort(dim=-1, stable=True).values[:, (band.shape[-1] - 1) // 2]
            tiny = torch.finfo(p4.dtype).eps
            return ((p4 - floor) / p4.clamp_min(tiny)).clamp(0.0, 1.0)

    def fourth_gain(self, spectrum: torch.Tensor) -> torch.Tensor:
        if self.mode == "lowpass4":
            return spectrum.real.new_ones(spectrum.shape[0])
        gain = torch.sigmoid(self.gate_logit)
        if self.mode == "static4":
            return gain.expand(spectrum.shape[0])
        return gain * self.noise_reliability(spectrum)

    def forward(self, raw: torch.Tensor, return_gate: bool = False):
        if raw.ndim != 3 or raw.shape[1:] != (PHASES, self.raw_points):
            raise ValueError(f"Expected [B,{PHASES},{self.raw_points}], got {tuple(raw.shape)}")
        spectrum = torch.fft.rfft(raw, n=self.raw_points, dim=-1, norm="ortho")
        # Construct a phase-independent real mask, retaining original complex
        # coefficients and hence relative phase at each retained harmonic.
        gain4 = self.fourth_gain(spectrum)
        mask = raw.new_zeros((raw.shape[0], self.raw_points // 2 + 1))
        mask[:, :4] = 1.0
        mask[:, 4] = gain4
        shaped = torch.fft.irfft(spectrum * mask.unsqueeze(1), n=self.raw_points,
                                 dim=-1, norm="ortho")
        return (shaped, gain4) if return_gate else shaped


class HPASFCNN(nn.Module):
    """The unchanged V2 A' CNN, with exactly one candidate spectral component."""
    def __init__(self, mode: str = "hpasf"):
        super().__init__()
        if mode not in METHODS:
            raise ValueError(mode)
        self.mode = mode
        self.shaping = (HarmonicShaping("lowpass3") if mode == "lowpass3"
                        else FourthHarmonicShaping(mode))
        self.normalize = MDNormalization()
        self.backbone = OriginalCNN()

    def forward(self, raw: torch.Tensor, return_gate: bool = False):
        if return_gate and self.mode != "lowpass3":
            shaped, gain4 = self.shaping(raw, return_gate=True)
        else:
            shaped, gain4 = self.shaping(raw), None
        input_128 = self.normalize(shaped)
        # Native V2 OriginalCNN.forward is identical to classifier(features(x)).
        logits = self.backbone.classifier(self.backbone.features(input_128))
        return (logits, gain4) if return_gate else logits


def counts_by_module(model: HPASFCNN) -> dict[str, int]:
    return {"cnn": sum(p.numel() for p in model.backbone.parameters()),
            "spectral": sum(p.numel() for p in model.shaping.parameters()),
            "total": sum(p.numel() for p in model.parameters())}


def smoke() -> None:
    torch.manual_seed(11)
    raw = torch.randn(2, 5, RAW_POINTS)
    labels = torch.tensor([0, 1])
    from v2_paifilter import paired_cross_entropy
    for mode in METHODS:
        network = HPASFCNN(mode)
        output = network(raw)
        assert output.shape == (2, CLASSES)
        loss = paired_cross_entropy(network, raw, raw + torch.randn_like(raw)*.1, labels)
        loss.backward()
        assert torch.isfinite(loss)
        if mode in ("static4", "hpasf"):
            assert network.shaping.gate_logit.grad is not None
            assert torch.isfinite(network.shaping.gate_logit.grad)
        print(mode, counts_by_module(network), f"loss={loss.item():.5f}")
    print("V4 HPASF SMOKE PASS")


if __name__ == "__main__":
    smoke()
