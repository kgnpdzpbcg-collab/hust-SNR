"""检验两种真实使用的噪声生成协议及MAT数值复现，不训练模型。"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_strong_snr import (gaussian_current, phase_snr, scaled_author_current,
                                 source_seed, save_checked)


@pytest.mark.parametrize("snr", [5, 0, -5])
def test_gaussian_phase_snr_is_exact_and_seeded(snr):
    """五相振幅不同、故障段近零时，也按整记录校准而非周期校准。"""
    t = np.arange(1000) / 1000
    clean = np.sin(2*np.pi*20*t[:, None]) * np.array([1, 2, 3, 4, 5])
    clean[500:, 0] = 0
    a = gaussian_current(clean, snr, 19)
    assert np.allclose(phase_snr(clean, a), snr, atol=1e-12, rtol=0)
    assert np.array_equal(a, gaussian_current(clean, snr, 19))
    assert not np.array_equal(a, gaussian_current(clean, snr, 20))


@pytest.mark.parametrize("snr", [5, 0, -5])
def test_scaled_author_noise_preserves_nonuniform_shape(snr):
    """作者噪声即使随时间变化，新档位仍保留其形态和每相SNR。"""
    rng = np.random.default_rng(21)
    clean = rng.standard_normal((1000, 5)) * np.arange(1, 6)
    noise = rng.standard_normal(clean.shape)
    noise[500:] *= .02
    noise *= np.sqrt(np.mean(clean**2, axis=0)/np.mean(noise**2, axis=0))/np.sqrt(10)
    original = clean + noise
    new = scaled_author_current(clean, original, snr)
    assert np.allclose(phase_snr(clean, new), snr, atol=1e-12, rtol=0)
    assert np.allclose(new-clean, (original-clean)*10**((10-snr)/20), atol=1e-14, rtol=0)
    assert np.array_equal(original, clean+noise)


def test_wrong_author_reference_is_rejected():
    """不能把其他噪声档位错当10 dB参考。"""
    clean = np.random.default_rng(1).standard_normal((1000, 5))
    with pytest.raises(ValueError, match="参考文件核算不符"):
        scaled_author_current(clean, gaussian_current(clean, 20, 2), 0)


def test_source_partitions_have_separate_fixed_random_streams():
    """固定源验证随机数跨重复执行一致，训练/保留集不复用同一随机流。"""
    values = [source_seed(part, "example", 5, 20261012) for part in ("train", "val", "source_test")]
    assert len(set(values)) == 3
    assert values[1] == source_seed("val", "example", 5, 20261012)
    assert values[1] != source_seed("val", "example", 0, 20261012)


def test_mat_column_layout_matches_original_loader_copy():
    """真实MAT的列连续布局与旧读取器copy后的行连续布局须逐值一致。"""
    original = np.asfortranarray(np.random.default_rng(17).standard_normal((10000, 6)))
    mat_currents = original[:, :5]
    old_currents = mat_currents.copy()
    assert mat_currents.flags.f_contiguous and old_currents.flags.c_contiguous
    assert np.array_equal(gaussian_current(mat_currents, 5, 38),
                          gaussian_current(old_currents, 5, 38))


def test_mat_round_trip_preserves_values_and_noncurrent_columns(tmp_path):
    """导出矩阵保持float64和原时间/角速度，无归一化或采样点删减。"""
    matrix = np.random.default_rng(12).standard_normal((100, 7))
    path = tmp_path / "SNR_5/example.mat"
    info = save_checked(path, "I_noisy", matrix)
    assert np.array_equal(loadmat(path)["I_noisy"], matrix)
    assert info["bytes"] == path.stat().st_size
