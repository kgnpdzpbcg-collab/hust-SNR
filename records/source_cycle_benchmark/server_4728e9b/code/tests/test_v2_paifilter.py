"""Self-contained unit tests for V2's phase-preserving harmonic CNN."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v2_paifilter import (HarmonicShaping, HarmonicCNN, MDNormalization, OriginalCNN,
                          RAW_POINTS, counts_by_module, paired_cross_entropy,
                          deterministic_linear_resample)


@pytest.mark.parametrize("sizes", [(4096, 128), (96, 2016)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_deterministic_interpolation_forward_and_backward(sizes, device):
    """核对两处真实插值的前向/梯度，并验证严格确定性CUDA反传重复一致。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("当前机器没有CUDA；服务器执行本项")
    original_mode = torch.are_deterministic_algorithms_enabled()
    original_warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(19)
        source_size, target_size = sizes
        original = torch.randn(2, 5, source_size, requires_grad=True)
        expected = torch.nn.functional.interpolate(
            original, size=target_size, mode="linear", align_corners=True)
        expected.square().mean().backward()
        gradients, outputs = [], []
        for _ in range(2):
            sample = original.detach().to(device).clone().requires_grad_()
            actual = deterministic_linear_resample(sample, target_size)
            actual.square().mean().backward()
            assert torch.allclose(actual.cpu(), expected.detach(), atol=5e-7, rtol=2e-6)
            assert torch.allclose(sample.grad.cpu(), original.grad, atol=1e-7, rtol=3e-5)
            gradients.append(sample.grad.detach().clone())
            outputs.append(actual.detach().clone())
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(gradients[0], gradients[1])
    finally:
        torch.use_deterministic_algorithms(original_mode, warn_only=original_warning)


def test_three_modes_shapes_and_parameters():
    for mode, expected in (("identity", 25620), ("lowpass3", 25620), ("paifilter", 25748)):
        model = HarmonicCNN(mode)
        assert counts_by_module(model)["total"] == expected
        logits, signal = model(torch.randn(3, 5, RAW_POINTS), True)
        assert logits.shape == (3, 6)
        assert signal.shape == (3, 5, 128)
        assert torch.isfinite(logits).all()


def test_identity_exactly_preserves_raw():
    raw = torch.randn(2, 5, RAW_POINTS)
    assert torch.equal(HarmonicShaping("identity")(raw), raw)


def test_lowpass_preserves_first_harmonics_and_suppresses_high():
    n = torch.arange(RAW_POINTS, dtype=torch.float32)
    low = 0.8 * torch.sin(2 * torch.pi * 2 * n / RAW_POINTS)
    high = 0.3 * torch.sin(2 * torch.pi * 40 * n / RAW_POINTS)
    x = (low + high).repeat(2, 5, 1)
    y = HarmonicShaping("lowpass3")(x)
    assert torch.max(torch.abs(y - low)).item() < 2e-5
    assert torch.allclose(y[0,0], y[0,3], atol=1e-5)


def test_paifilter_has_no_phase_rotation_or_amplification():
    filt = HarmonicShaping("paifilter")
    with torch.no_grad():
        filt.raw_attenuation.fill_(2.)
    g = filt.gains(torch.randn(2,5,RAW_POINTS))
    assert float(g[0].detach()) == 1.0
    assert bool((g >= 0).all() and (g <= 1).all())
    n = torch.arange(RAW_POINTS, dtype=torch.float32)
    x = torch.sin(2 * torch.pi * 4 * n/RAW_POINTS + 0.4).view(1,1,-1).repeat(1,5,1)
    y = filt(x)
    expect = torch.exp(torch.tensor(-2.)) * x
    assert torch.allclose(y, expect, atol=1e-5)


def test_paifilter_gradient_and_optimizer_step():
    torch.manual_seed(6)
    model = HarmonicCNN("paifilter")
    model.eval() # keep BN deterministic for nonzero-spectrum gradient
    x = torch.randn(8,5,RAW_POINTS)
    y = x + 0.2*torch.randn_like(x)
    labels = torch.tensor([0,1,2,3,4,5,0,1])
    loss = paired_cross_entropy(model, x, y, labels)
    loss.backward()
    grad = model.shaping.raw_attenuation.grad
    assert grad is not None and bool(torch.isfinite(grad).all())
    assert float(grad.abs().sum()) > 1e-8
    before = model.shaping.raw_attenuation.detach().clone()
    optim = torch.optim.SGD(model.parameters(), lr=0.25)
    optim.step()
    assert not torch.equal(before, model.shaping.raw_attenuation.detach())


def test_normalization_low_current_branch_and_range():
    n = torch.linspace(0,2*torch.pi, RAW_POINTS)
    a = torch.sin(n)
    near_zero = 0.05*torch.sin(n)
    x = torch.stack([a,near_zero,a,a,a]).unsqueeze(0)
    norm = MDNormalization()(x)
    assert norm.shape==(1,5,128)
    assert norm[0,1].abs().max().item() < 0.001
    assert norm[0,0].abs().max().item() > 0.95
    assert norm.abs().max().item() <= 1.00001


def test_original_backbone_state_dict_compatible_with_v1():
    # Gate0/V1 backbone names are 'features.*' and 'classifier.*'.
    backbone = OriginalCNN()
    assert list(backbone.state_dict())[:2] == ["features.0.weight", "features.0.bias"]
    assert backbone.classifier[3].out_features==6
    assert backbone.classifier[1].out_features==50


def test_invalid_shapes_and_modes():
    with pytest.raises(ValueError): HarmonicShaping("spectrogram")
    with pytest.raises(ValueError): HarmonicShaping("paifilter", 511)
    with pytest.raises(ValueError): HarmonicShaping("paifilter")(torch.randn(2,1,1024))
