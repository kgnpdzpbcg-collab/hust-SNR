"""V5 NA-HSF invariants, differentiability and deterministic CPU/CUDA behavior."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from v5_nahsf import (NAHSFCNN, NoiseAwareHarmonicFusion, parameter_counts,
                      PHASES, RAW_POINTS)
from v2_paifilter import HarmonicShaping, paired_cross_entropy
from v4_hpasf import HPASFCNN


def signal(amp: dict[int, float], batches=2):
    t = torch.arange(RAW_POINTS) / RAW_POINTS
    out = sum(v * torch.cos(2*math.pi*k*t) for k,v in amp.items())
    return out.view(1,1,-1).repeat(batches, 5, 1).float()


def amp_at(t, k):
    return torch.fft.rfft(t, dim=-1, norm='ortho')[:, :, k].abs().mean().item()


def test_parameter_count_and_midpoint_initialization():
    m = NAHSFCNN()
    assert parameter_counts(m) == {'cnn': 25620, 'spectral': 41, 'total': 25661}
    raw = torch.randn(3, 5, RAW_POINTS)
    y, a, z = m(raw, return_descriptors=True)
    assert y.shape == (3,6) and z.shape == (3,3)
    assert torch.equal(a, torch.full_like(a, 0.5))


def test_midpoint_is_exact_lp3_lp4_average_before_normalization():
    raw = torch.randn(2, 5, RAW_POINTS)
    filter3 = HarmonicShaping('lowpass3')(raw)
    filter4 = torch.fft.rfft(raw, norm='ortho')
    f = torch.zeros_like(filter4)
    f[:, :, :5] = filter4[:, :, :5]
    filter4 = torch.fft.irfft(f, n=RAW_POINTS, norm='ortho')
    with torch.no_grad():
        output, gate = NoiseAwareHarmonicFusion()(raw, return_gate=True)
    assert torch.allclose(output, 0.5*filter3 + 0.5*filter4, atol=2e-6, rtol=2e-6)
    assert torch.all(gate == .5)


def test_shared_real_gain_preserves_harmonic_phase():
    torch.manual_seed(31)
    t = torch.arange(RAW_POINTS)/RAW_POINTS
    raw = torch.stack([torch.cos(2*math.pi*4*t + 0.37*p) + 0.8*torch.cos(2*math.pi*t+.17*p)
                       for p in range(5)]).unsqueeze(0)
    model = NoiseAwareHarmonicFusion()
    out, alpha = model(raw, return_gate=True)
    orig_spec = torch.fft.rfft(raw, norm='ortho')
    out_spec = torch.fft.rfft(out, norm='ortho')
    assert torch.allclose(out_spec[:, :, 4], orig_spec[:, :, 4]*alpha[:, None], atol=5e-6)
    assert torch.allclose(out_spec[:, :, 1], orig_spec[:, :, 1], atol=5e-6)
    assert (out_spec[:, :, 6:].abs() < 2e-5).all()


def test_spectral_descriptors_are_finite_and_bounded_for_zero_signal():
    model = NoiseAwareHarmonicFusion()
    raw = torch.zeros(2, PHASES, RAW_POINTS)
    out, alpha, z = model(raw, return_descriptors=True)
    assert torch.isfinite(out).all() and torch.isfinite(alpha).all()
    assert torch.isfinite(z).all()
    assert z[:, 0].min() >= 0 and z[:, 0].max() <= 1
    assert z[:, 2].min() >= 0 and z[:, 2].max() <= 1
    assert (z[:, 1].abs() <= 1).all()


def test_noise_descriptor_increases_with_high_frequency_energy():
    clean = signal({1:1,2:.3,4:.15},batches=1)
    extra = signal({12:.5,13:.7,14:.4,15:.6,16:.3,17:.8,18:.4,19:.5,
                    20:.5,21:.7,22:.4,23:.6,24:.3,25:.8,26:.4,27:.5,
                    28:.4,29:.6,30:.3}, batches=1)
    m = NoiseAwareHarmonicFusion()
    _, _, z = m(torch.cat((clean, clean+extra),dim=0), return_descriptors=True)
    assert z[1, 2] > z[0, 2]
    assert z[1, 1] < z[0, 1]


def test_gradient_flows_through_filter_and_mlp():
    torch.manual_seed(8)
    model = NAHSFCNN()
    x = torch.randn(4, 5, RAW_POINTS)
    y = torch.tensor([0,1,2,3])
    opt = torch.optim.Adam(model.parameters(),lr=.001)
    loss = paired_cross_entropy(model,x,x+.1*torch.randn_like(x),y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    weight = model.shaping.gate[-1].weight
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert weight.grad.abs().max().item() > 0
    assert model.backbone.features[0].weight.grad.abs().max().item() > 0
    before = weight.detach().clone()
    opt.step()
    assert not torch.equal(before, weight.detach())
    # second step unlocks gradients into first MLP layer
    opt.zero_grad(set_to_none=True)
    loss = paired_cross_entropy(model,x,x+.1*torch.randn_like(x),y)
    loss.backward()
    assert model.shaping.gate[0].weight.grad is not None
    assert model.shaping.gate[0].weight.grad.abs().max().item() > 0


def test_cnn_initialization_matches_v4_for_same_seed():
    for seed in (42,43,44):
        torch.manual_seed(seed)
        old = HPASFCNN('lowpass4')
        torch.manual_seed(seed)
        new = NAHSFCNN()
        for x,y in zip(old.backbone.parameters(), new.backbone.parameters()):
            assert torch.equal(x,y)


def test_invalid_shape_and_nan_rejected():
    model = NoiseAwareHarmonicFusion()
    with pytest.raises(ValueError):
        model(torch.randn(2,5,128))
    bad = torch.randn(2,5,RAW_POINTS)
    bad[0,0,0] = float('nan')
    with pytest.raises(ValueError):
        model(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_strict_determinism_and_gate_gradient():
    torch.use_deterministic_algorithms(True)
    try:
        model = NAHSFCNN().cuda()
        data = torch.randn(3,5,RAW_POINTS,device='cuda')
        y = torch.tensor([0,1,2],device='cuda')
        a = model(data, return_gate=True)[1]
        b = model(data, return_gate=True)[1]
        assert torch.equal(a,b)
        loss = paired_cross_entropy(model,data,data+.1*torch.randn_like(data),y)
        loss.backward()
        assert torch.isfinite(model.shaping.gate[-1].weight.grad).all()
    finally:
        torch.use_deterministic_algorithms(False)
