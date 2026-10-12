"""Pure CPU deterministic tests: V4 harmonic prior, gate and A' parity."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v4_hpasf import (HPASFCNN, FourthHarmonicShaping, METHODS, RAW_POINTS,
                      counts_by_module, INITIAL_GAIN4)
from v2_paifilter import HarmonicShaping, paired_cross_entropy


def waveform(freqs: dict[int, float], batch: int = 2) -> torch.Tensor:
    t = torch.arange(RAW_POINTS, dtype=torch.float32) / RAW_POINTS
    signal = sum(amp * torch.cos(2 * math.pi * k * t) for k, amp in freqs.items())
    return signal.view(1, 1, -1).expand(batch, 5, -1).clone()


def harmonic_amp(current, k):
    z = torch.fft.rfft(current, dim=-1, norm="ortho")
    return z[..., k].abs().mean().item()


def test_four_modes_have_frozen_cnn_and_only_one_spectral_parameter():
    for method in METHODS:
        model = HPASFCNN(method)
        counts = counts_by_module(model)
        assert counts["cnn"] == 25620
        assert counts["spectral"] == (1 if method in ("static4", "hpasf") else 0)
        assert counts["total"] == counts["cnn"] + counts["spectral"]
        assert model(torch.randn(2, 5, RAW_POINTS)).shape == (2, 6)


def test_lowpass3_exact_v2_implementation():
    raw = torch.randn(2, 5, RAW_POINTS)
    newer = HPASFCNN("lowpass3")
    older = HarmonicShaping("lowpass3")
    assert torch.equal(newer.shaping(raw), older(raw))


def test_fixed_fourth_passes_order4_and_rejects_high_order():
    signal = waveform({1: 1, 2: 0.4, 3: 0.3, 4: 0.5, 7: 1.0})
    out, g4 = FourthHarmonicShaping("lowpass4")(signal, return_gate=True)
    assert torch.equal(g4, torch.ones_like(g4))
    assert abs(harmonic_amp(out, 4) / harmonic_amp(signal, 4) - 1) < 1e-5
    assert harmonic_amp(out, 7) / harmonic_amp(signal, 7) < 1e-4


def test_static_and_hpasf_begin_near_strong_lowpass3_prior():
    for mode in ("static4", "hpasf"):
        m = FourthHarmonicShaping(mode)
        assert float(torch.sigmoid(m.gate_logit).detach()) == pytest.approx(INITIAL_GAIN4, rel=1e-5)
        sample = waveform({1: 1.0, 4: 0.5})
        out, gain = m(sample, return_gate=True)
        assert 0 <= gain.min() <= gain.max() <= INITIAL_GAIN4 + 1e-6
        assert harmonic_amp(out, 1) / harmonic_amp(sample, 1) == pytest.approx(1,abs=1e-5)
        assert harmonic_amp(out, 4) <= 0.051 * harmonic_amp(sample, 4)


def test_reliability_is_per_sample_and_decreases_as_noise_floor_rises():
    clean = waveform({1: 1, 4: 0.15}, batch=1)
    noisy = clean + waveform({12: 0.6, 13: 0.5, 14: 0.4, 15: 0.6, 16: 0.5,
                               17: 0.4, 18: 0.6, 19: 0.5, 20: 0.4,
                               21: 0.6, 22: 0.5, 23: 0.4, 24: 0.6,
                               25: 0.5, 26: 0.4, 27: 0.6, 28: 0.5,
                               29: 0.4, 30: 0.6}, batch=1)
    model = FourthHarmonicShaping("hpasf")
    _, g = model(torch.cat([clean, noisy]), return_gate=True)
    assert g.shape == (2,)
    assert g[0] > 0.04
    assert g[1] < 0.005
    assert (g >= 0).all() and (g <= INITIAL_GAIN4 + 1e-6).all()


def test_source_gains_do_not_depend_on_snr_labels_or_clean_reference():
    assert not hasattr(FourthHarmonicShaping("hpasf"), "target_labels")
    raw = torch.randn(3, 5, RAW_POINTS)
    m = FourthHarmonicShaping("hpasf")
    _, gates = m(raw, return_gate=True)
    _, gates_repeat = m(raw.clone(), return_gate=True)
    assert torch.equal(gates, gates_repeat)
    assert torch.all((gates >= 0) & (gates <= 1))


def test_static_and_adaptive_gate_trainable_with_ce_and_optimizer():
    torch.manual_seed(1)
    base = waveform({1: 1.0, 3: .2, 4: .5},batch=2)
    for mode in ("static4", "hpasf"):
        model = HPASFCNN(mode)
        opt = torch.optim.Adam(model.parameters(),lr=.001)
        prior = model.shaping.gate_logit.detach().clone()
        noisy = base + .02*torch.randn_like(base)
        loss = paired_cross_entropy(model,base,noisy,torch.tensor([1,2]))
        loss.backward()
        grad = model.shaping.gate_logit.grad
        assert grad is not None and torch.isfinite(grad) and grad.abs().item() > 0
        opt.step()
        assert not torch.equal(prior,model.shaping.gate_logit.detach())


def test_median_noise_floor_matches_explicit_harmonic_band():
    spec = torch.zeros((2,5,RAW_POINTS//2+1),dtype=torch.complex64)
    spec[:,:,4] = 2+0j           # p4=4
    spec[:,:,12:31] = 1+0j       # median floor=1
    r = FourthHarmonicShaping.noise_reliability(spec)
    assert torch.allclose(r,torch.full((2,),.75))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        HPASFCNN("unknown")
    with pytest.raises(ValueError):
        FourthHarmonicShaping("static4", initial_gain4=0)
    with pytest.raises(ValueError):
        FourthHarmonicShaping("hpasf")(torch.randn(2,5,128))
