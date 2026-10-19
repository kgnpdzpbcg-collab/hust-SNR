"""CPU tests for the V3 paper-informed Jung/Hang adaptations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest
import torch

from jung_hang_models_v3 import (
    HaarDWTLLPool,
    HangWCNN,
    JungModifiedCNN,
    add_white_noise_snr_preserve_quiet,
    current_window_to_jung_input,
    current_window_to_normalized_vsd_image,
    five_phase_vsd,
)


def _balanced_five_phase(points=1001):
    time = np.linspace(0.0, 2.0 * np.pi, points, endpoint=True)
    theta = np.deg2rad(np.arange(5) * 72.0)
    current = np.stack([np.cos(time - angle) for angle in theta], axis=1)
    return time, current


def test_jung_model_shape_parameter_count_and_gradients():
    torch.manual_seed(1)
    model = JungModifiedCNN(in_channels=5, num_classes=6)
    logits = model(torch.randn(4, 5, 128))
    assert logits.shape == (4, 6)
    assert sum(p.numel() for p in model.parameters()) == 7110
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_jung_input_shared_scale_preserves_interphase_amplitude_ratios():
    time = np.linspace(0.0, 2.0 * np.pi, 1001)
    theta = np.deg2rad(np.arange(5) * 72.0)
    amplitudes = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    currents = np.stack(
        [amplitudes[p] * np.cos(time - theta[p]) for p in range(5)], axis=1
    )
    value = current_window_to_jung_input(time, currents, time[0], time[-1], points=128)
    peak_by_phase = np.max(np.abs(value), axis=1)
    assert value.shape == (5, 128)
    assert np.allclose(peak_by_phase / peak_by_phase[0], amplitudes, atol=0.035)
    assert np.isfinite(value).all()


def test_haar_ll_pool_matches_the_paper_sum_filter_and_halves_shape():
    pool = HaarDWTLLPool()
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    out = pool(x)
    expected = torch.tensor([[[[10.0, 18.0], [42.0, 50.0]]]])
    assert torch.equal(out, expected)
    assert list(pool.parameters()) == []
    assert out.shape == (1, 1, 2, 2)


def test_hang_wcnn_uses_40x40_single_plane_input_and_has_finite_gradients():
    torch.manual_seed(2)
    model = HangWCNN(in_channels=1, num_classes=6, image_size=40)
    logits = model(torch.randn(4, 1, 40, 40))
    assert logits.shape == (4, 6)
    assert sum(p.numel() for p in model.parameters()) == 63294
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.randn(4, 1, 64, 64))


def test_five_phase_vsd_fundamental_projection_is_correct():
    time = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    theta = np.deg2rad(np.arange(5) * 72.0)
    currents = np.stack([np.cos(time - angle) for angle in theta], axis=1)
    transformed = five_phase_vsd(currents)
    assert transformed.shape == (4, len(time))
    assert np.allclose(transformed[0], np.cos(time), atol=1e-10)
    assert np.allclose(transformed[1], np.sin(time), atol=1e-10)
    assert np.max(np.abs(transformed[2:])) < 1e-10


def test_hang_trajectory_is_instantaneously_normalized_and_scale_invariant():
    time, currents = _balanced_five_phase()
    a = current_window_to_normalized_vsd_image(
        time, currents, time[0], time[-1], points=128, image_size=40
    )
    b = current_window_to_normalized_vsd_image(
        time, 7.0 * currents, time[0], time[-1], points=128, image_size=40
    )
    assert a.shape == (1, 40, 40)
    assert np.isfinite(a).all()
    assert a.sum() > 0
    assert np.array_equal(a, b)


def test_quiet_current_gate_preserves_zero_intervals_and_calibrates_snr():
    fs = 10000.0
    time = np.arange(10000) / fs
    omega = np.full_like(time, 2.0 * np.pi * 50.0)
    theta = np.deg2rad(np.arange(5) * 72.0)
    currents = np.stack([np.sin(2 * np.pi * 50 * time - a) for a in theta], axis=1)
    quiet = (time >= 0.35) & (time <= 0.65)
    currents[quiet, 0] = 0.0
    noisy = add_white_noise_snr_preserve_quiet(
        time, currents, omega, 20.0, np.random.default_rng(17)
    )
    noise = noisy - currents
    assert np.all(noise[quiet[200:-200] if False else quiet, 0] == 0.0)
    for phase in range(5):
        signal_power = np.mean(currents[:, phase] ** 2)
        noise_power = np.mean(noise[:, phase] ** 2)
        measured = 10 * np.log10(signal_power / noise_power)
        assert measured == pytest.approx(20.0, abs=1e-9)


def test_zero_signal_adapters_return_finite_zeros():
    time = np.linspace(0.0, 1.0, 1001)
    currents = np.zeros((len(time), 5), dtype=np.float64)
    jung = current_window_to_jung_input(time, currents, 0.1, 0.9)
    hang = current_window_to_normalized_vsd_image(time, currents, 0.1, 0.9)
    assert not jung.any()
    assert not hang.any()
