"""CPU-only unit tests for Jung/Hang V4 fidelity-controlled baselines."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest
import torch

from jung_hang_models_v4 import jung_input, hang_vsd_image, add_noise
from compare_jung_hang_v4 import _two_cycle_bounds, build_inputs, smoke


CFG = dict(jung_cycles=1, jung_scale="shared", hang_planes="dual",
           hang_center="none", hang_normalization="shared")


def _current(samples=1001, turns=1, third=0.0, dc=0.0):
    time = np.linspace(0, 2 * np.pi * turns, samples)
    phase = 2 * np.pi * np.arange(5) / 5
    current = np.stack([np.sin(time - a) + third * np.sin(3 * (time - a))
                        for a in phase], axis=1)
    current[:, 0] += dc
    return time, current


def test_jung_scales_preserve_or_remove_between_phase_amplitude_ratio():
    time, x = _current()
    x[:, 3] *= 0.1
    shared = jung_input(time, x, time[0], time[-1], scale_mode="shared")
    per_phase = jung_input(time, x, time[0], time[-1], scale_mode="per_phase")
    assert shared.shape == per_phase.shape == (5, 128)
    assert np.max(np.abs(shared[3])) < 0.2 * np.max(np.abs(shared[0]))
    assert np.max(np.abs(per_phase[3])) == pytest.approx(1.0, abs=0.005)
    assert np.max(np.abs(per_phase[0])) == pytest.approx(1.0, abs=0.005)


def test_two_cycle_uses_only_adjacent_same_label_cycles():
    time, signal = _current(samples=2001, turns=2)
    raw = {"rec": (time, signal, np.zeros_like(time))}
    meta = [{"record_id": "rec", "label": 1, "start": 0.0, "end": 2 * np.pi},
            {"record_id": "rec", "label": 1, "start": 2 * np.pi, "end": 4 * np.pi}]
    c = dict(CFG, jung_cycles=2)
    x = build_inputs("jung", meta, raw, c)
    assert x.shape == (2, 5, 256)
    assert np.isfinite(x).all()
    assert np.allclose(x[0], x[1], atol=1e-6)


def test_two_cycle_fails_at_split_gap_or_fault_boundary():
    meta = [{"record_id": "rec", "label": 1, "start": 0.0, "end": 1.0},
            {"record_id": "rec", "label": 1, "start": 1.15, "end": 2.15}]
    with pytest.raises(ValueError, match="Two-cycle input unavailable"):
        _two_cycle_bounds(meta)
    changed_label = [{"record_id": "rec", "label": 1, "start": 0.0, "end": 1.0},
                     {"record_id": "rec", "label": 2, "start": 1.0, "end": 2.0}]
    with pytest.raises(ValueError, match="Two-cycle input unavailable"):
        _two_cycle_bounds(changed_label)


def test_hang_same_capacity_for_single_and_dual_vsd():
    t, x = _current(third=0.25)
    single = hang_vsd_image(t, x, t[0], t[-1], planes="fundamental")
    dual = hang_vsd_image(t, x, t[0], t[-1], planes="dual")
    assert single.shape == dual.shape == (2, 40, 40)
    assert np.any(single[0])
    assert not np.any(single[1])
    assert np.any(dual[1])


def test_hang_center_choice_and_normalization():
    t, x = _current(dc=1.0)
    raw = hang_vsd_image(t, x, t[0], t[-1], center_mode="none")
    centered = hang_vsd_image(t, x, t[0], t[-1], center_mode="phase_mean")
    unit = hang_vsd_image(t, x, t[0], t[-1],
                          normalization="instantaneous", quiet_ratio=0.05)
    assert np.isfinite(raw).all() and np.isfinite(centered).all()
    assert raw.shape == unit.shape == (2, 40, 40)
    assert not np.array_equal(raw, centered)


def test_hang_source_inputs_preserve_two_channels():
    t, x = _current()
    raw = {"rec": (t, x, np.zeros_like(t))}
    meta = [{"record_id": "rec", "label": 0, "start": 0.0, "end": 2 * np.pi}]
    values = build_inputs("hang", meta, raw, CFG)
    assert values.shape == (1, 2, 40, 40)
    assert np.isfinite(values).all()


def test_plain_awgn_has_target_record_snr():
    t, x = _current()
    omega = np.ones_like(t) * 100.0
    y = add_noise(t, x, omega, 20.0, np.random.default_rng(123), mode="awgn")
    noise = y - x
    signal_power = np.mean(x ** 2, axis=0)
    noise_power = np.mean(noise ** 2, axis=0)
    np.testing.assert_allclose(10 * np.log10(signal_power / noise_power),
                               np.full(5, 20.0), atol=1e-9)
    assert np.isfinite(y).all()


def test_finite_gradients_and_model_shapes():
    result = smoke(CFG)
    assert result["jung_output"] == result["hang_output"] == (3, 6)
    from compare_jung_hang_v4 import get_model
    for method, shape in [("jung", (4, 5, 128)), ("hang", (4, 2, 40, 40))]:
        model = get_model(method, CFG)
        out = model(torch.randn(shape))
        loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
