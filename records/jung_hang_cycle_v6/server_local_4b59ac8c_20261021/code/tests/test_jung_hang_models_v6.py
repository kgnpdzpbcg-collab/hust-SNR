"""V6 高保真输入的 CPU 测试。

重点是两条"回归护栏"：
  * Hang 的归一化必须是标量常数——故障窗的轨迹点不能全落在单位圆上。
    逐采样时刻归一化会让该比例恒等于 1，这正是 V5 崩溃的直接原因。
  * Jung 的原生 THD 必须在整数个电周期上统计——纯正弦的 THD 必须接近 0。
    不裁剪时矩形窗泄漏会把 THD 抬到约 2.6%，目标 THD+N=3% 时几乎不注噪。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pytest
import torch

from compare_jung_hang_v6 import build_inputs, get_model, image_audit
from jung_hang_models_v3 import _draw_polyline, _interpolate_current_window, five_phase_vsd
from jung_hang_models_v6 import (
    BLOCKED_PHASES, add_awgn, blocked_phase_mask, fundamental_thd,
    hang_trajectory, jung_thdn_augment, jung_window, record_reference,
)

FS = 10000.0
F0 = 40.0
ANGLE = np.deg2rad(np.arange(5) * 72.0)
T = np.arange(10000) / FS
OMEGA = np.full(T.shape, 2.0 * np.pi * F0)
AMPLITUDE = 10.0
OFFSET = 0.25


def _healthy():
    """平衡五相正弦，带一个已知的直流偏置。"""
    return AMPLITUDE * np.cos(2.0 * np.pi * F0 * T[:, None] - ANGLE[None, :]) + OFFSET


def _half_cycle_blocked(currents, onset=0.5):
    """A 相正半周被完全阻断，对应 OLS/OP 类故障的非线性特征。

    onset 之后才发生故障，与真实数据协议一致：0.10–0.45 s 是故障前健康段，
    直流偏置和健康矢量模长都只能在这段上估计。
    """
    blocked = currents.copy()
    positive = (np.cos(2.0 * np.pi * F0 * T) >= 0.0) & (T >= onset)
    blocked[positive, 0] = OFFSET
    return blocked


def _window(start=0.5):
    return start, start + 1.0 / F0


def _per_sample_normalized_image(time, currents, start, end, dc, points=128, size=40):
    """V5 的写法：逐采样时刻除以自身模长，仅用于对照测试。"""
    _, segment = _interpolate_current_window(time, currents - dc, start, end, points)
    components = five_phase_vsd(segment.T)
    magnitude = np.hypot(components[0], components[1])
    valid = magnitude > 1e-12
    x = np.zeros_like(magnitude)
    y = np.zeros_like(magnitude)
    x[valid] = components[0][valid] / magnitude[valid]
    y[valid] = components[1][valid] / magnitude[valid]
    return _draw_polyline(x, y, size, valid=valid)[None]


def test_record_reference_recovers_dc_and_healthy_modulus():
    currents = _healthy()
    dc, scale, used = record_reference(T, currents)
    assert used is True
    assert np.allclose(dc, OFFSET, atol=1e-2)
    assert scale == pytest.approx(AMPLITUDE, rel=0.01)


def test_jung_window_is_one_cycle_with_shared_scale():
    currents = _healthy()
    dc, _, _ = record_reference(T, currents)
    start, end = _window()
    window = jung_window(T, currents, start, end, dc, points=200)
    assert window.shape == (5, 200)
    assert window.dtype == np.float32
    assert float(np.max(np.abs(window))) == pytest.approx(1.0, rel=1e-6)
    # 五相共用同一尺度：平衡正弦下各相峰值应相同，且保留相对相位。
    peaks = np.max(np.abs(window), axis=1)
    assert np.allclose(peaks, peaks[0], rtol=0.05)


def test_hang_scalar_normalization_keeps_radial_shape():
    """健康窗是单位圆；半周阻断窗必须出现明显小于 1 的半径。"""
    healthy = _healthy()
    dc, scale, _ = record_reference(T, healthy)
    start, end = _window()

    healthy_image = hang_trajectory(T, healthy, start, end, dc, scale)
    assert healthy_image.shape == (2, 40, 40)
    healthy_fraction = image_audit(healthy_image[None], [0])["on_unit_circle_fraction_by_channel"][0]
    assert healthy_fraction == pytest.approx(1.0, abs=1e-6)

    faulted = _half_cycle_blocked(healthy)
    fault_image = hang_trajectory(T, faulted, start, end, dc, scale)
    fraction = image_audit(fault_image[None], [0])["on_unit_circle_fraction_by_channel"][0]
    assert fraction < 0.9, f"故障轨迹仍几乎全在单位圆上: {fraction}"

    per_sample = _per_sample_normalized_image(T, faulted, start, end, dc)
    assert image_audit(per_sample[None], [0])["on_unit_circle_fraction"] == pytest.approx(1.0, abs=1e-6)


def test_hang_images_differ_between_healthy_and_fault():
    healthy = _healthy()
    dc, scale, _ = record_reference(T, healthy)
    start, end = _window()
    a = hang_trajectory(T, healthy, start, end, dc, scale)
    b = hang_trajectory(T, _half_cycle_blocked(healthy), start, end, dc, scale)
    assert not np.array_equal(a, b)
    assert image_audit(np.stack((a, b)), [0, 1])["duplicate_groups"] == 0


def test_fundamental_thd_recovers_known_harmonic_and_rejects_leakage():
    pure = _healthy()
    dc, _, _ = record_reference(T, pure)
    _, thd_pure = fundamental_thd(T, pure, OMEGA, dc)
    # 纯正弦：裁剪到整数周期后不应有可测谐波，泄漏底噪远低于 0.5%。
    assert np.all(thd_pure < 5e-3), f"存在频谱泄漏: {thd_pure}"

    with_harmonic = pure.copy()
    with_harmonic += 0.4 * np.cos(3.0 * 2.0 * np.pi * F0 * T[:, None] - ANGLE[None, :])
    fundamental, thd = fundamental_thd(T, with_harmonic, OMEGA, dc)
    assert np.allclose(fundamental, AMPLITUDE / np.sqrt(2.0), rtol=0.02)
    assert np.allclose(thd, 0.04, rtol=0.10)


def test_thdn_augment_matches_equation_5_and_respects_native_thd():
    base = _healthy()
    with_harmonic = base + 0.4 * np.cos(3.0 * 2.0 * np.pi * F0 * T[:, None] - ANGLE[None, :])
    dc, _, _ = record_reference(T, with_harmonic)

    rng = np.random.default_rng(0)
    target = 6.0
    noisy = jung_thdn_augment(T, with_harmonic, OMEGA, target, rng, dc=dc, limiter=False)
    fundamental, native = fundamental_thd(T, with_harmonic, OMEGA, dc)
    expected = fundamental * np.sqrt((target / 100.0) ** 2 - native ** 2)
    measured = np.sqrt(np.mean((noisy - with_harmonic) ** 2, axis=0))
    assert np.allclose(measured, expected, rtol=0.05)

    # 目标低于原生 THD 时，式 (5) 给出 sigma=0，必须完全不加噪。
    quiet = jung_thdn_augment(T, with_harmonic, OMEGA, 3.0, rng, dc=dc, limiter=False)
    assert np.array_equal(quiet, with_harmonic)


def test_blocked_phase_mask_only_applies_to_blocked_classes():
    healthy = _healthy()
    currents = _half_cycle_blocked(healthy)
    dc, _, _ = record_reference(T, currents)
    metadata = [{"record_id": "r", "label": 0, "start": 0.0, "end": 0.9}]

    assert BLOCKED_PHASES[1] == () and BLOCKED_PHASES[4] == ()
    mask_ous = blocked_phase_mask(T, currents, OMEGA, [dict(metadata[0], label=1)], dc)
    assert mask_ous.all(), "OUS 类不应加限幅"

    mask_ols = blocked_phase_mask(T, currents, OMEGA, [dict(metadata[0], label=2)], dc)
    inside = (T >= 0.505) & (T < 0.9)
    cosine = np.cos(2.0 * np.pi * F0 * T)
    # 只断言远离过零点的区段，并跳过故障起始后 5 ms：
    # 限幅器在电流接近零时本来就会阻断噪声，半周切换处少量采样被屏蔽是
    # 物理上正确的行为；起始跳变沿附近的卷积窗还会跨越健康段，需留出过渡带。
    assert not mask_ols[inside & (cosine >= 0.5), 0].any(), "明显被阻断的区段仍允许加噪"
    assert mask_ols[inside & (cosine <= -0.5), 0].all(), "明显导通的区段被误屏蔽"
    # 其余四相未被阻断，任何时刻都应允许加噪。
    assert mask_ols[:, 1:].all()


def test_add_awgn_reaches_requested_snr():
    currents = _healthy()
    noisy = add_awgn(currents, 15.0, np.random.default_rng(1))
    signal_power = np.mean(currents ** 2, axis=0)
    noise_power = np.mean((noisy - currents) ** 2, axis=0)
    achieved = 10.0 * np.log10(signal_power / noise_power)
    assert np.allclose(achieved, 15.0, atol=0.5)


def _config():
    return {
        "reference_window_s": [0.10, 0.45],
        "jung_cycles": 1,
        "hang_cycles": 1,
        "jung_points_per_cycle": 200,
        "hang_points": 128,
        "hang_image_size": 40,
        "hang_planes": "dual",
        "hang_phase_angles_deg": [0, 72, 144, 216, 288],
    }


def test_build_inputs_keeps_every_anchor_for_both_models():
    currents = _healthy()
    raw = {"r": (T, currents, OMEGA)}
    meta = [{"record_id": "r", "label": 0, "start": 0.50 + i / F0, "end": 0.51 + i / F0}
            for i in range(3)]

    jung, jung_positions = build_inputs("jung", meta, raw, _config())
    hang, hang_positions = build_inputs("hang", meta, raw, _config())

    assert jung.shape == (3, 5, 200) and hang.shape == (3, 2, 40, 40)
    np.testing.assert_array_equal(jung_positions, np.arange(3))
    np.testing.assert_array_equal(hang_positions, np.arange(3))
    assert np.isfinite(jung).all() and np.isfinite(hang).all()


def test_model_shapes_and_parameter_counts():
    config = _config()
    jung = get_model("jung", config)
    hang = get_model("hang", config)
    assert sum(p.numel() for p in jung.parameters()) == 7110
    assert sum(p.numel() for p in hang.parameters()) == 63330
    assert tuple(jung(torch.randn(2, 5, 200)).shape) == (2, 6)
    assert tuple(hang(torch.randn(2, 2, 40, 40)).shape) == (2, 6)
