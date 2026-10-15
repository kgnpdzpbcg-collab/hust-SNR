"""CPU-only prototype checks. No original MAT files, server, or Target-Holdout."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hust_a0_wavelet_attention import (
    A0MultiPhaseMWACNN, SharedRMSNormalization, haar_dwt,
    a0_paired_loss,
)
from hust_b0_harmonic_recovery import (
    B0HarmonicRecoveryNet, HarmonicRecovery, harmonic_recovery_loss,
    b0_paired_loss,
)
from hust_prototypes_experiment import (
    METHODS, SEEDS, TRAIN_NOISE, VAL_NOISE, CONDITIONS,
    build_model, get_config,
)
from generate_extreme_dev_snr import (
    full_record_phase_snr, scale_author_residual,
)

torch.set_num_threads(2)


def test_haar_orthonormality_and_shape():
    x = torch.randn(2, 7, 128)
    lo, hi = haar_dwt(x)
    assert lo.shape == hi.shape == (2, 7, 64)
    assert torch.allclose(lo.square().sum() + hi.square().sum(),
                          x.square().sum(), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        haar_dwt(torch.rand(2, 5, 127))


def test_a0_preserves_phase_ratios_before_pooling():
    x = torch.zeros(1, 5, 4096)
    x[:, 0] = 2.0
    x[:, 1] = -1.0
    y = SharedRMSNormalization()(x)
    assert y.shape == (1, 5, 512)
    assert torch.allclose(y[:, 0], -2 * y[:, 1], atol=1e-6)
    assert torch.count_nonzero(y[:, 2:]) == 0


def test_a0_forward_and_backward():
    torch.manual_seed(42)
    model = A0MultiPhaseMWACNN()
    clean = torch.randn(2, 5, 4096)
    noisy = clean + 0.2 * torch.randn_like(clean)
    labels = torch.tensor([1, 3])
    logits, features = model(clean, return_features=True)
    assert logits.shape == (2, 6)
    assert features.shape == (2, 96)
    loss = a0_paired_loss(model, clean, noisy, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.layers[0].analysis.fam[0].weight.grad is not None
    assert torch.isfinite(model.layers[0].analysis.fam[0].weight.grad).all()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(clean), model(clean))
    with pytest.raises(ValueError):
        model(torch.randn(2, 6, 4096))


def test_b0_harmonic_output_and_dc_validity():
    torch.manual_seed(42)
    module = HarmonicRecovery(max_harmonic=8)
    raw = torch.randn(2, 5, 4096)
    restored, spectrum = module(raw, return_spectrum=True)
    assert restored.shape == raw.shape
    assert spectrum.shape == (2, 5, 9)
    assert torch.all(spectrum[:, :, 0].imag == 0)
    assert torch.isfinite(restored).all()
    ref = torch.fft.rfft(raw, dim=-1, norm="ortho")[:, :, :9]
    # Initial model is close to identity for retained harmonics.
    assert torch.allclose(spectrum, ref * torch.sigmoid(torch.tensor(4.)),
                          rtol=3e-4, atol=1e-5)


def test_b0_recovery_supervision_and_gradients():
    torch.manual_seed(43)
    net = B0HarmonicRecoveryNet(max_harmonic=8)
    clean = torch.randn(2, 5, 4096)
    noisy = clean + 0.1 * torch.randn_like(clean)
    labels = torch.tensor([0, 4])
    logits, spec = net(noisy, return_recovery=True)
    assert logits.shape == (2, 6)
    assert spec.shape == (2, 5, 9)
    rec = harmonic_recovery_loss(spec, clean)
    assert rec.item() >= 0 and torch.isfinite(rec)
    loss, parts = b0_paired_loss(net, clean, noisy, labels, .2)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(parts["ce"]) and torch.isfinite(parts["rec"])
    assert net.restorer.delta.weight.grad is not None
    assert torch.isfinite(net.restorer.delta.weight.grad).all()
    assert net.backbone.classifier.weight.grad is not None
    assert torch.isfinite(net.backbone.classifier.weight.grad).all()


def test_b0_zero_open_phase_and_bad_inputs():
    raw = torch.randn(1, 5, 4096)
    raw[:, 0] = 0
    model = HarmonicRecovery()
    restored = model(raw)
    assert torch.isfinite(restored).all()
    with pytest.raises(ValueError):
        model(torch.randn(1, 5, 128))
    with pytest.raises(ValueError):
        harmonic_recovery_loss(torch.randn(1, 5, 9), raw)


def test_extreme_snr_exact_and_no_input_mutation():
    rng = np.random.default_rng(9)
    clean = rng.normal(size=(10031, 5))
    error = rng.normal(size=clean.shape)
    error *= np.sqrt(np.mean(clean**2, axis=0) /
                     np.mean(error**2, axis=0)) / (10 ** (10/20))
    author = clean + error
    frozen = author.copy()
    assert np.max(np.abs(full_record_phase_snr(clean, author) - 10.)) < 1e-10
    for snr in (-5, -10):
        generated = scale_author_residual(clean, author, snr)
        assert np.max(np.abs(full_record_phase_snr(clean, generated) - snr)) < 1e-10
    assert np.array_equal(author, frozen)
    with pytest.raises(ValueError):
        scale_author_residual(clean, author, 0)


def test_frozen_config_and_model_factory():
    c = get_config()
    assert tuple(c["seeds"]) == SEEDS == (42, 43, 44)
    assert tuple(c["methods"]) == METHODS
    assert tuple(c["train_noise_db"]) == TRAIN_NOISE
    assert tuple(c["val_noise_db"]) == VAL_NOISE
    assert tuple(c["main_target_conditions_db"]) == CONDITIONS
    assert c["source_only"] and not c["final_holdout_evaluated"]
    assert isinstance(build_model("a0"), A0MultiPhaseMWACNN)
    assert isinstance(build_model("b0"), B0HarmonicRecoveryNet)
    with pytest.raises(ValueError):
        build_model("resnet_lp4")
