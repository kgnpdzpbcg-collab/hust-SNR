"""A1/B1 source-only CPU tests; GPU determinism and real MAT are server checks."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hust_a1_reliability_wavelet import (
    A1NoiseAwareWaveletCNN, ReliabilityDWA, RobustPhaseScale, a1_paired_loss)
from hust_b1_phase_harmonic import (
    B1PhaseAwareRecoveryNet, PhaseAwareHarmonicRestorer,
    shared_harmonic_scale, phase_strength, b1_paired_loss,
    b1_recovery_objective)
from hust_a1_b1_experiment import (
    METHODS, SEEDS, TRAIN_NOISE, VAL_NOISE, CONDITIONS,
    get_config, build_model, select_source_checkpoint)

torch.set_num_threads(2)


def test_frozen_research_protocol():
    c = get_config()
    assert c["protocol"] == "hust-a1-b1-source-only-v1"
    assert METHODS == ("a1", "b1")
    assert SEEDS == (42, 43, 44)
    assert TRAIN_NOISE == (10, 15, 20, 25, 30, 35, 40)
    assert VAL_NOISE == (0, 5, 10, 20, 30)
    assert CONDITIONS == (5, 0, -5, -10)
    assert c["final_holdout_evaluated"] is False and c["source_only"] is True
    assert isinstance(build_model("a1"), A1NoiseAwareWaveletCNN)
    assert isinstance(build_model("b1"), B1PhaseAwareRecoveryNet)
    with pytest.raises(ValueError):
        build_model("a0")


def test_a1_normalization_is_global_scale_invariant_and_keeps_open_phase_zero():
    n = 4096
    phase = torch.arange(n, dtype=torch.float32) * 2 * math.pi / n
    raw = torch.stack([
        torch.sin(phase), 0.5 * torch.sin(phase),
        torch.sin(phase + .3), torch.zeros_like(phase),
        torch.sin(phase + .7)], dim=0)[None]
    module = RobustPhaseScale()
    z1 = module(raw)
    z2 = module(raw * 3.7)
    assert z1.shape == (1, 5, 512)
    assert torch.allclose(z1, z2, atol=1e-5, rtol=1e-5)
    assert z1[:, 3].abs().max() == 0


def test_a1_reliability_gate_sees_high_detail_rms_and_never_amplifies():
    # Equal low coefficients / signed detail means, hundredfold difference
    # in detail ENERGY. Original A0 FAM could not discriminate these cases.
    t = torch.ones(1, 5, 64)
    e = torch.tensor([1, -1], dtype=t.dtype).repeat(64).view(1, 1, 128)
    x_lo = (t.repeat_interleave(2, -1) + .1 * e)
    x_hi = (t.repeat_interleave(2, -1) + 1.0 * e)
    gate = ReliabilityDWA(5, dropout=0.0)
    ylo, g_lo = gate(x_lo, return_gate=True)
    yhi, g_hi = gate(x_hi, return_gate=True)
    assert ylo.shape == yhi.shape == (1, 10, 64)
    assert torch.allclose(ylo[:, :5], yhi[:, :5], atol=1e-6)
    assert (g_lo > g_hi).all().item()
    assert ((g_lo >= 0) & (g_lo <= 1)).all().item()
    assert ((g_hi >= 0) & (g_hi <= 1)).all().item()
    assert torch.allclose(ylo[:, :5], (2.0 ** .5) * t)


def test_a1_forward_backward_and_no_label_at_inference():
    torch.manual_seed(42)
    model = A1NoiseAwareWaveletCNN()
    x = torch.randn(2, 5, 4096)
    y = torch.tensor([0, 4])
    out, feats = model(x, return_features=True)
    assert out.shape == (2, 6)
    assert feats.shape == (2, 96)
    loss = a1_paired_loss(model, x, x+.2*torch.randn_like(x), y)
    loss.backward()
    assert torch.isfinite(loss).item()
    grad = model.blocks[0].wavelet.adjustment[-1].weight.grad
    assert grad is not None and grad.isfinite().all().item()
    with pytest.raises(ValueError):
        model(torch.randn(2, 6, 4096))


def test_b1_robust_common_scale_and_weak_phase_structure():
    t = torch.arange(4096, dtype=torch.float32) * 2*math.pi/4096
    phase = torch.stack([
        torch.sin(t), torch.sin(t+.2),
        torch.sin(t+.4), torch.zeros_like(t),
        torch.sin(t+.8)], dim=0)[None]
    spec = torch.fft.rfft(phase, dim=-1, norm="ortho")[:, :, :5]
    scale = shared_harmonic_scale(spec)
    assert torch.all(scale > 0).item()
    ratio = phase_strength(spec)
    assert ratio.shape == (1, 5)
    assert ratio[:, 3].max().item() < 1e-4
    assert ratio[:, [0, 1, 2, 4]].min().item() > .1


def test_b1_gain_and_complex_phase_are_bounded_and_no_harmonic_invention():
    torch.manual_seed(42)
    model = PhaseAwareHarmonicRestorer()
    x = torch.randn(2, 5, 4096)
    restored, spec, variance, gain = model(x, return_details=True)
    assert restored.shape == x.shape
    assert spec.shape == variance.shape == gain.shape == (2, 5, 5)
    assert torch.isfinite(restored).all().item()
    assert ((gain > 0) & (gain < 1)).all().item()
    assert (variance > 0).all().item()
    observed = torch.fft.rfft(x, dim=-1, norm="ortho")[:, :, :5]
    assert torch.allclose(spec, observed * gain, rtol=1e-5, atol=1e-5)
    assert spec[:, :, 0].imag.abs().max().item() == 0
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 128))


def test_b1_noisy_to_clean_loss_all_terms_and_recovery_gradients():
    torch.manual_seed(43)
    net = B1PhaseAwareRecoveryNet()
    clean = torch.randn(3, 5, 4096)
    noisy = clean + .2 * torch.randn_like(clean)
    labels = torch.tensor([1, 3, 5])
    logits, spec, variance, gain = net(noisy, return_recovery=True)
    assert logits.shape == (3, 6)
    assert spec.shape == (3, 5, 5)
    rec, components = b1_recovery_objective(noisy, clean, spec, variance)
    assert rec.item() > 0 and torch.isfinite(rec).item()
    assert set(components) == {"complex", "phase", "noise"}
    total, parts = b1_paired_loss(net, clean, noisy, labels)
    total.backward()
    assert torch.isfinite(total).item()
    assert torch.isfinite(parts["restore"]).item()
    for param in (net.restorer.log_noise_order_scale,
                  net.restorer.adjustment[-1].weight,
                  net.backbone.stem[0].weight,
                  net.backbone.classifier.weight):
        assert param.grad is not None and torch.isfinite(param.grad).all().item()


def test_b1_auxiliary_physical_signal_preserves_absent_phase_after_normalization():
    x = torch.zeros(1, 5, 4096)
    t = torch.arange(4096, dtype=torch.float32) * 2*math.pi/4096
    x[:, 1] = torch.sin(t)
    x[:, 2] = torch.sin(t+.2)
    x[:, 3] = torch.sin(t+.5)
    x[:, 4] = torch.sin(t+.8)
    physical = B1PhaseAwareRecoveryNet._physical_view(x)
    assert physical.shape == (1, 5, 128)
    assert physical[:, 0].abs().max().item() == 0
    assert physical[:, 1].abs().max().item() > .1


def test_source_checkpoint_selector_handles_f1_saturation_without_target():
    hist = [
        {"epoch":1,"clean_f1":1.0,"noise_mean_f1":1.0,"noise_mean_ce":.53},
        {"epoch":2,"clean_f1":1.0,"noise_mean_f1":1.0,"noise_mean_ce":.08},
        {"epoch":3,"clean_f1":1.0,"noise_mean_f1":1.0,"noise_mean_ce":.10},
    ]
    assert select_source_checkpoint(hist)["epoch"] == 2
    # F1 remains dominant; source CE is only the tie breaker.
    hist[2]["noise_mean_f1"] = .999
    assert select_source_checkpoint(hist)["epoch"] == 2
    with pytest.raises(ValueError):
        select_source_checkpoint([])


def test_strict_no_oracle_and_clean_target_validation():
    import inspect
    assert "target" not in inspect.signature(
        A1NoiseAwareWaveletCNN.forward).parameters
    assert "clean" not in inspect.signature(
        PhaseAwareHarmonicRestorer.forward).parameters
    with pytest.raises(ValueError):
        b1_recovery_objective(torch.randn(2, 5, 128),
                              torch.randn(2, 5, 4096),
                              torch.zeros(2, 5, 5, dtype=torch.complex64),
                              torch.ones(2, 5, 5))


def test_b1_gain_invariant_to_circular_cycle_shift():
    """Only restorer gains are invariant; the downstream CNN need not be."""
    torch.manual_seed(123)
    x = torch.randn(2, 5, 4096)
    net = PhaseAwareHarmonicRestorer()
    with torch.no_grad():
        _, spectrum, _, gain = net(x, return_details=True)
        shifted = x.roll(512, dims=-1)
        _, spectrum_shifted, _, shifted_gain = net(
            shifted, return_details=True)
    assert torch.allclose(gain, shifted_gain, rtol=1e-5, atol=1e-5)
    rotated_observed = torch.fft.rfft(shifted, dim=-1, norm="ortho")[:, :, :5]
    assert torch.allclose(spectrum_shifted, rotated_observed * shifted_gain,
                          rtol=1e-5, atol=1e-5)


def test_b1_noise_proxy_rises_with_random_highband_noise():
    """Check noisy input-only descriptor, not calibrated target SNR."""
    torch.manual_seed(321)
    t = torch.arange(4096, dtype=torch.float32) * 2*math.pi/4096
    clean = torch.stack(
        [torch.sin(t + .2*i) for i in range(5)], dim=0)[None]
    noise = torch.randn_like(clean)
    low = clean + 0.05*noise
    high = clean + 0.35*noise
    restorer = PhaseAwareHarmonicRestorer()
    def descriptor(x):
        ft = torch.fft.rfft(x, dim=-1, norm="ortho")
        return restorer.noise_proxy(ft, shared_harmonic_scale(ft))
    assert descriptor(high).mean() > 10*descriptor(low).mean()
