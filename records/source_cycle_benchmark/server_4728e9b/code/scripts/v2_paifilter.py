"""V2: FilterNet-PaiFilter-inspired phase-preserving harmonic shaping for HUST.

Architecture only: raw 4096-point five-phase electrical cycle -> shared real FFT
harmonic gain (identity/fixed/learnable) -> differentiable HUST MD centering
and low-current normalization -> 128-point resampling -> *unchanged* V1 CNN.

Reference: Yi et al., FilterNet: Harnessing Frequency Filters for Time Series
Forecasting, NeurIPS 2024; official PaiFilter.py uses a learned circular
convolution via rFFT(w).  Our direct REAL harmonic gates are a task-specific
adaptation (not a faithful reproduction of the forecasting model).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

RAW_POINTS = 4096
CNN_POINTS = 128
PHASES = 5
CLASSES = 6
TRAINABLE_BANDS = 128
EXACT_LOW_HARMONICS = 32
METHODS = ("identity", "lowpass3", "paifilter")


def deterministic_linear_resample(values: torch.Tensor, size: int) -> torch.Tensor:
    """沿最后一维线性插值，保持align_corners=True的端点和采样位置。

    输入为任意前导维度的浮点张量，输出只改变最后一维长度。本项目
    用于96→2016个频率控制点和4096→128个电流采样点。显式索引与
    权重避开CUDA的非确定性upsample反传；梯度仍流向输入/滤波参数。
    与原算子存在浮点运算顺序带来的末位舍入差异，不增加可训练参数。
    """
    source_size = values.shape[-1]
    if source_size < 2 or size < 2:
        raise ValueError("线性插值的输入和输出长度均须至少为2")
    position = torch.arange(size, device=values.device, dtype=values.dtype)
    position = (position * ((source_size - 1) / (size - 1))).clamp(max=source_size - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=source_size - 1)
    weight = position - left
    return values.index_select(-1, left) * (1 - weight) + values.index_select(-1, right) * weight


class HarmonicShaping(nn.Module):
    """Phase-preserving real rFFT gains, shared across five phase currents.

    identity: exact no-op, for pipeline-controlled M1 reimplementation.
    lowpass3: exact hard selection k=0..3 (frequency *order*, not Hz).
    paifilter: trainable non-amplifying 0..1 gain for k=1..N/2.

    Soft nonnegative attenuation w=exp(-relu(raw)) allows an (almost) identity
    start with nonzero gradients. The k=0 DC gain is always exactly 1.
    This operation is not learned denoising nor does it use target input SNR.
    """
    def __init__(self, mode: str = "paifilter", raw_points: int = RAW_POINTS):
        super().__init__()
        if mode not in METHODS:
            raise ValueError(f"Unknown mode: {mode}")
        if raw_points < 128 or raw_points % 2 != 0:
            raise ValueError("raw_points must be an even integer >= 128")
        self.mode = mode
        self.raw_points = raw_points
        if mode == "paifilter":
            # +1e-3 is intentional: passes nearly untouched at initialization
            # while retaining nonzero gradient toward *more* attenuation.
            # First 32 harmonics keep independent gains; 96 band anchors
            # interpolate smoothly across higher harmonics to avoid learning
            # 2048 independent knobs from just 738 simulation train cycles.
            self.raw_attenuation = nn.Parameter(torch.full((TRAINABLE_BANDS,), 1e-3))
        else:
            self.register_parameter("raw_attenuation", None)

    def gains(self, ref: torch.Tensor) -> torch.Tensor:
        count = self.raw_points // 2 + 1
        if self.mode == "identity":
            return ref.new_ones(count)
        if self.mode == "lowpass3":
            gains = ref.new_zeros(count)
            gains[:4] = 1.0
            return gains
        # A scalar gain for each harmonic order, *not* a complex phase rotation.
        low = self.raw_attenuation[:EXACT_LOW_HARMONICS]
        high = self.raw_attenuation[EXACT_LOW_HARMONICS:].view(1, 1, -1)
        expanded_high = deterministic_linear_resample(
            high, self.raw_points // 2 - EXACT_LOW_HARMONICS).view(-1)
        attenuation = torch.relu(torch.cat((low, expanded_high)))
        return torch.cat((ref.new_ones(1), torch.exp(-attenuation)), dim=0)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or raw.shape[1:] != (PHASES, self.raw_points):
            raise ValueError(f"Expected [B,{PHASES},{self.raw_points}], got {tuple(raw.shape)}")
        if self.mode == "identity":
            # Avoid an unnecessary FFT round trip when verifying baseline parity.
            return raw
        spectra = torch.fft.rfft(raw, n=self.raw_points, dim=-1, norm="ortho")
        filtered = spectra * self.gains(raw).to(spectra.dtype)[None, None, :]
        return torch.fft.irfft(filtered, n=self.raw_points, dim=-1, norm="ortho")


class MDNormalization(nn.Module):
    """Differentiable approximation of prepare.preprocess's exact MD rule.

    On raw 4096-point uniform per-cycle samples: midpoint of extrema;
    if centered mean abs < 0.2 then divisor=100, else max centered abs;
    then linear resample to CNN's 128-point input.

    Differentiable almost everywhere, except the low-current branch boundary
    (matching the fixed original MD preprocessing rule).
    """
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        center = (raw.amax(dim=-1, keepdim=True) + raw.amin(dim=-1, keepdim=True)) * 0.5
        centered = raw - center
        low = centered.abs().mean(dim=-1, keepdim=True) < 0.2
        maximum = centered.abs().amax(dim=-1, keepdim=True)
        divisor = torch.where(low, torch.full_like(maximum, 100.), maximum.clamp_min(1e-8))
        scaled = centered / divisor
        return deterministic_linear_resample(scaled, CNN_POINTS)


class OriginalCNN(nn.Module):
    """Same 25,620 parameters and layer topology as the frozen V1 CNN."""
    def __init__(self):
        super().__init__()
        layers = []
        for inc, out, kernel in ((5, 16, 16), (16, 32, 4), (32, 32, 4)):
            layers.extend([nn.Conv1d(inc, out, kernel), nn.BatchNorm1d(out),
                           nn.LeakyReLU(.01), nn.MaxPool1d(2)])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(352, 50),
                                        nn.LeakyReLU(.01), nn.Linear(50, CLASSES))

    def forward(self, x):
        return self.classifier(self.features(x))


class HarmonicCNN(nn.Module):
    """Pre-normalization harmonic shaping + original CNN classifier.

    The only trainable addition to the reference CNN is 128 real
    harmonic attenuation parameters in mode='paifilter'.
    """
    def __init__(self, mode: str = "paifilter", raw_points: int = RAW_POINTS):
        super().__init__()
        self.shaping = HarmonicShaping(mode, raw_points)
        self.normalize = MDNormalization()
        self.backbone = OriginalCNN()

    def forward(self, raw: torch.Tensor, return_signal: bool = False):
        shaped = self.shaping(raw)
        signal = self.normalize(shaped)
        logits = self.backbone(signal)
        return (logits, signal) if return_signal else logits


def paired_cross_entropy(model: HarmonicCNN, clean: torch.Tensor,
                         noisy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Exact V1 M1 classification objective, without V1 consistency loss."""
    logits = model(torch.cat((clean, noisy), dim=0))
    n = labels.shape[0]
    return 0.5 * (F.cross_entropy(logits[:n], labels) +
                  F.cross_entropy(logits[n:], labels))


def counts_by_module(model: HarmonicCNN) -> dict:
    return {"cnn": sum(p.numel() for p in model.backbone.parameters()),
            "spectral": sum(p.numel() for p in model.shaping.parameters()),
            "total": sum(p.numel() for p in model.parameters())}


def demo() -> None:
    torch.manual_seed(11)
    raw = torch.randn(4, PHASES, RAW_POINTS)
    labels = torch.tensor([0, 1, 2, 3])
    for mode in METHODS:
        model = HarmonicCNN(mode)
        pred, signal = model(raw, return_signal=True)
        assert pred.shape == (4, CLASSES)
        assert signal.shape == (4, PHASES, CNN_POINTS)
        loss = paired_cross_entropy(model, raw, raw + 0.1 * torch.randn_like(raw), labels)
        loss.backward()
        if mode == "paifilter":
            grad = model.shaping.raw_attenuation.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        print(mode, counts_by_module(model), f"CE={loss.item():.5f}")
    print("V2 ARCHITECTURE SMOKE PASSED")


if __name__ == "__main__":
    demo()
