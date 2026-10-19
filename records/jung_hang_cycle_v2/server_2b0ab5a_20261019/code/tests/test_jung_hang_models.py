"""CPU-only tests for the Jung/Hang approximate reproductions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest
import torch

from jung_hang_models import (
    HaarDWTLLPool,
    HangWCNN,
    JungModifiedCNN,
    current_window_to_vsd_images,
    five_phase_vsd,
)
from compare_jung_hang import add_white_noise_snr, classification_metrics, select_checkpoint


def test_jung_model_shape_parameters_and_gradients():
    torch.manual_seed(1)
    model = JungModifiedCNN()
    x = torch.randn(4, 5, 128)
    logits = model(x)
    assert logits.shape == (4, 6)
    assert sum(p.numel() for p in model.parameters()) == 7110
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_haar_ll_pool_is_fixed_and_halves_spatial_shape():
    pool = HaarDWTLLPool()
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    out = pool(x)
    expected = torch.tensor([[[[10.0, 18.0], [42.0, 50.0]]]])
    assert torch.equal(out, expected)
    assert list(pool.parameters()) == []
    assert out.shape == (1, 1, 2, 2)


def test_hang_wcnn_shape_parameters_and_gradients():
    torch.manual_seed(2)
    model = HangWCNN()
    x = torch.randn(4, 2, 64, 64)
    logits = model(x)
    assert logits.shape == (4, 6)
    assert sum(p.numel() for p in model.parameters()) == 123234
    assert list(model.pool1.parameters()) == []
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_five_phase_vsd_separates_fundamental_from_third_plane():
    t = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    theta = np.deg2rad(np.arange(5) * 72.0)
    currents = np.stack([np.cos(t - angle) for angle in theta], axis=1)
    transformed = five_phase_vsd(currents)
    assert transformed.shape == (4, len(t))
    assert np.allclose(transformed[0], np.cos(t), atol=1e-10)
    assert np.allclose(transformed[1], np.sin(t), atol=1e-10)
    assert np.max(np.abs(transformed[2:])) < 1e-10


def test_vsd_trajectory_images_are_fixed_shape_finite_and_not_independently_scaled():
    t = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    theta = np.deg2rad(np.arange(5) * 72.0)
    currents = np.stack([np.cos(t - angle) for angle in theta], axis=1)
    image = current_window_to_vsd_images(t, currents, t[0], t[-1], points=128, image_size=64)
    assert image.shape == (2, 64, 64)
    assert np.isfinite(image).all()
    assert image[0].sum() > 0


def test_snr_noise_augmentation_is_reproducible_and_shape_preserving():
    t = np.linspace(0.0, 4.0 * np.pi, 1000)
    signal = np.stack([np.sin(t + p) for p in np.arange(5)], axis=1)
    a = add_white_noise_snr(signal, 20, np.random.default_rng(99))
    b = add_white_noise_snr(signal, 20, np.random.default_rng(99))
    assert a.shape == signal.shape
    assert np.array_equal(a, b)
    noise = a - signal
    measured_snr = 10 * np.log10(np.mean(signal ** 2) / np.mean(noise ** 2))
    assert measured_snr == pytest.approx(20.0, abs=0.35)


def test_checkpoint_rule_clean_guard_then_noisy_f1():
    history = [
        {"epoch": 1, "clean_val_macro_f1": 0.95, "noisy_val_mean_macro_f1": 0.50},
        {"epoch": 2, "clean_val_macro_f1": 0.94, "noisy_val_mean_macro_f1": 0.70},
        {"epoch": 3, "clean_val_macro_f1": 0.93, "noisy_val_mean_macro_f1": 0.90},
    ]
    assert select_checkpoint(history)["epoch"] == 2


def test_metrics_separate_normal_false_alarms_from_fault_misses():
    truth = [0, 0, 1, 1, 2, 3, 4, 5]
    pred = [0, 2, 0, 1, 2, 3, 4, 5]
    scores = classification_metrics(truth, pred)
    assert scores["normal_false_alarms"] == 1
    assert scores["normal_false_alarm_rate"] == pytest.approx(0.5)
    assert scores["missed_fault_as_normal"] == 1
    assert len(scores["confusion_matrix"]) == 6
