"""Regression tests for the five source-only paper B1 ablations.

CPU synthetic cycles only. These tests do NOT evaluate measured Dev, do NOT
read any target labels, and do NOT claim trained ablation performance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from b1_paper_ablation_models import (
    A1, A2, A3, A4, A5, METHODS,
    A1FixedLP4DualView, A2FixedNoiseProxy, A3NoInterphaseDescriptors,
    A4MDOnly, A5NoStrengthLoss, MaskInterphaseDescriptors,
    build_model, calibrate_a2_from_source, paired_loss, pretrain_restorer,
    recovery_loss,
)
from hust_b1_phase_harmonic import B1PhaseAwareRecoveryNet
from v2_paifilter import RAW_POINTS, PHASES


@pytest.fixture(scope="module", autouse=True)
def small_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 2))
    yield
    torch.set_num_threads(old)


def cycles(batch: int = 2) -> torch.Tensor:
    """Coherent phase shifted cycles, one zeroed current, plus small noise."""
    generator = torch.Generator().manual_seed(812)
    t = 2 * torch.pi * torch.arange(RAW_POINTS) / RAW_POINTS
    x = torch.stack([torch.sin(t + i * 2 * torch.pi / PHASES)
                     + 0.15 * torch.sin(2 * t + i * .31)
                     for i in range(PHASES)])[None].repeat(batch, 1, 1)
    x[:, -1, :] = 0.01 * torch.sin(t)
    return x + 0.03 * torch.randn(x.shape, generator=generator)


def test_five_explicit_models_and_input_shapes():
    assert METHODS == (A1, A2, A3, A4, A5)
    for method in METHODS:
        model = build_model(method).eval()
        if method == A2:
            with pytest.raises(RuntimeError, match="not calibrated"):
                model(cycles(1))
            model.restorer.calibrate_source(cycles(2))
        with torch.no_grad():
            logits = model(cycles(2))
        assert logits.shape == (2, 6)
        assert torch.isfinite(logits).all()
    with pytest.raises(ValueError, match="Unknown"):
        build_model("b1")
    assert A1FixedLP4DualView().backbone.stem[0].in_channels == 10
    assert A4MDOnly().backbone.stem[0].in_channels == 5


def test_a1_is_exact_fixed_harmonic_projection_and_has_no_restorer_parameters():
    model = A1FixedLP4DualView().eval()
    raw = cycles()
    with torch.no_grad():
        restored, spec, variance, gain = model.restorer(raw, return_details=True)
    spectrum = torch.fft.rfft(raw, dim=-1, norm="ortho")
    recovered_spectrum = torch.fft.rfft(restored, dim=-1, norm="ortho")
    torch.testing.assert_close(spec, spectrum[..., :5])
    torch.testing.assert_close(recovered_spectrum[..., :5], spectrum[..., :5],
                               atol=1e-5, rtol=1e-5)
    assert float(recovered_spectrum[..., 5:].abs().max()) < 1e-5
    assert variance.count_nonzero() == 0
    assert bool((gain == 1).all())
    assert sum(p.numel() for p in model.restorer.parameters()) == 0


def test_a2_requires_source_only_calibration_and_persists_buffer(monkeypatch):
    model = A2FixedNoiseProxy()
    with pytest.raises(RuntimeError):
        model.restorer.noise_proxy(torch.empty(1, PHASES, 8), torch.ones(1))
    source_clean = cycles(3).numpy()
    source_noisy = source_clean + np.random.default_rng(9).normal(
        0, .2, source_clean.shape).astype(np.float32)
    def fake_augmentation(raw, metadata, seed, epoch, noise_levels):
        assert raw == "SOURCE_TRAIN_ONLY"
        assert metadata == ["TRAIN_META"]
        assert seed == 42 and epoch == 1
        assert tuple(noise_levels) == (10, 15, 20, 25, 30, 35, 40)
        return source_noisy
    import v2_experiment
    monkeypatch.setattr(v2_experiment, "fresh_noisy_source", fake_augmentation)
    calibrate_a2_from_source(model, "SOURCE_TRAIN_ONLY", ["TRAIN_META"], 42,
                             (10, 15, 20, 25, 30, 35, 40))
    p = model.restorer.fixed_proxy.item()
    assert np.isfinite(p) and p > 0
    spectrum = torch.fft.rfft(cycles(), dim=-1, norm="ortho")
    proxy = model.restorer.noise_proxy(spectrum, torch.ones(2))
    assert proxy.shape == (2, 5)
    torch.testing.assert_close(proxy, torch.full_like(proxy, p))
    assert "restorer.fixed_proxy" in model.state_dict()
    loaded = A2FixedNoiseProxy()
    loaded.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(loaded.restorer.fixed_proxy, model.restorer.fixed_proxy)
    assert torch.isfinite(loaded(cycles(1))).all()


def test_a3_masks_exact_feature_channels_without_removing_trainable_params():
    mask = MaskInterphaseDescriptors()
    z = torch.randn(3, 26, 5, requires_grad=True)
    out = mask(z)
    torch.testing.assert_close(out[:, :5], z[:, :5])
    torch.testing.assert_close(out[:, 15:], z[:, 15:])
    assert out[:, 5:15].count_nonzero() == 0
    out.square().sum().backward()
    assert z.grad is not None
    assert z.grad[:, 5:15].count_nonzero() == 0
    assert z.grad[:, :5].abs().sum() > 0
    with pytest.raises(ValueError, match="layout"):
        mask(torch.randn(1, 25, 5))
    baseline = B1PhaseAwareRecoveryNet()
    ablated = A3NoInterphaseDescriptors()
    n = lambda x: sum(p.numel() for p in x.parameters())
    assert n(baseline) == n(ablated)
    assert isinstance(ablated.restorer.adjustment[0], MaskInterphaseDescriptors)


def test_a4_only_changes_stem_channel_count_and_retains_restoration():
    baseline, a4 = B1PhaseAwareRecoveryNet(), A4MDOnly()
    assert type(baseline.restorer) is type(a4.restorer)
    assert baseline.backbone.stem[0].in_channels == 10
    assert a4.backbone.stem[0].in_channels == 5
    for name in ("layers", "pool", "classifier"):
        assert (list(getattr(baseline.backbone, name).state_dict().keys())
                == list(getattr(a4.backbone, name).state_dict().keys()))
    with torch.no_grad():
        logits, spec, variance, gain = a4(cycles(), return_recovery=True)
    assert logits.shape == (2, 6)
    assert spec.shape == variance.shape == gain.shape == (2, PHASES, 5)


def test_a5_is_identical_to_b1_at_inference_and_differs_only_in_loss():
    baseline = B1PhaseAwareRecoveryNet().eval()
    a5 = A5NoStrengthLoss().eval()
    a5.load_state_dict(baseline.state_dict(), strict=True)
    raw = cycles()
    with torch.no_grad():
        torch.testing.assert_close(baseline(raw), a5(raw), atol=0, rtol=0)
        _, rec, variance, _ = baseline(raw, return_recovery=True)
    normal, _ = recovery_loss(A4, raw, raw, rec, variance)
    removed, parts = recovery_loss(A5, raw, raw, rec, variance)
    torch.testing.assert_close(normal-removed, 0.4 * parts["phase"], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("method", METHODS)
def test_training_loss_finite_and_gradient_flows(method):
    model = build_model(method)
    x = cycles(2)
    if method == A2:
        model.restorer.calibrate_source(x + .3 * torch.randn_like(x))
    noisy = x + 0.2 * torch.randn_like(x)
    loss, parts = paired_loss(method, model, x, noisy, torch.tensor([1, 4]))
    assert torch.isfinite(loss).item()
    assert "ce" in parts
    assert ("restore" in parts) == (method != A1)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in model.parameters())
    if method == A3:
        assert model.restorer.adjustment[1].weight.grad is not None


def test_runner_frozen_source_only_config_and_no_full_baseline_override():
    from b1_paper_ablations import (
        METHODS as RUN_METHODS, SEEDS, CONDITIONS, configuration, OUT, PROTOCOL,
    )
    cfg = configuration()
    assert RUN_METHODS == METHODS == tuple(cfg["methods"])
    assert SEEDS == (42, 44) == tuple(cfg["seeds"])
    assert CONDITIONS == (5, 0, -5, -10) == tuple(cfg["target_conditions"])
    assert cfg["target_training"] is False
    assert cfg["target_checkpoint_selection"] is False
    assert cfg["final_holdout_evaluated"] is False
    assert "source_cycle_benchmark" not in str(OUT)
    assert cfg["protocol"] == PROTOCOL


def test_source_only_pretrain_uses_correct_a5_loss_and_skips_a1(monkeypatch):
    from types import SimpleNamespace
    import v2_experiment
    x = cycles(2).numpy()
    def synthetic_noise(raw, meta, seed, epoch, noise_levels):
        assert raw == "SOURCE_TRAIN_ONLY"
        assert seed == 42 and epoch == 1
        return x + 0.1 * np.ones_like(x)
    monkeypatch.setattr(v2_experiment, "fresh_noisy_source", synthetic_noise)
    config = {"learning_rate": 1e-3, "batch_size": 2,
              "b1_pretrain_epochs": 1, "train_noise_db": [10]}
    assert pretrain_restorer(A1, A1FixedLP4DualView(),
                             "SOURCE_TRAIN_ONLY", [], x, 42, config,
                             torch.device("cpu")) == []
    model = A5NoStrengthLoss()
    hist = pretrain_restorer(A5, model, "SOURCE_TRAIN_ONLY", [], x, 42,
                             config, torch.device("cpu"))
    assert len(hist) == 1 and np.isfinite(hist[0])
