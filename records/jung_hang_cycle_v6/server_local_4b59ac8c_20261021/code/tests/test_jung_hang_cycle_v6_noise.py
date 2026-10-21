"""V6运行入口的噪声口径回归：验证使用SNR，Jung训练保留THD+N。"""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import jung_hang_cycle_v6 as runner
from jung_hang_models_v6 import fundamental_thd, record_reference


@pytest.fixture
def source_record():
    """10kHz健康五相正弦，不依赖实际源/目标数据文件。"""
    t = np.arange(10000) / 10000
    phase = np.arange(5) * 2 * np.pi / 5
    currents = 10 * np.cos(2 * np.pi * 40 * t[:, None] - phase)
    omega = np.full(len(t), 2 * np.pi * 40)
    config_path = Path(__file__).resolve().parents[1] / 'configs/jung_hang_v6_v1.json'
    config = json.loads(config_path.read_text(encoding='utf8'))
    return t, currents, omega, config


@pytest.mark.parametrize('method', ['jung', 'hang'])
@pytest.mark.parametrize('snr', [0, 5, 30])
def test_validation_uses_requested_snr_db(source_record, method, snr):
    """直接测量新增噪声功率，防止把dB数值传成THD+N百分比。"""
    t, clean, omega, config = source_record
    result = runner.noise_records({'r': (t, clean, omega)}, 42, 1,
                                 method=method, meta=[], config=config, snr=snr)
    actual_time, noisy, actual_speed = result['r']
    np.testing.assert_array_equal(actual_time, t)
    np.testing.assert_array_equal(actual_speed, omega)
    power = np.mean(clean ** 2, axis=0)
    noise_power = np.mean((noisy - clean) ** 2, axis=0)
    assert np.all(noise_power > 0)
    achieved = 10 * np.log10(power / noise_power)
    # 加噪函数不对随机实现二次定标；固定10k样本的实际功率允许小幅统计波动。
    np.testing.assert_allclose(achieved, np.full(5, snr), atol=0.25, rtol=0)


def test_jung_training_keeps_three_percent_thdn(source_record):
    """验证分支修复不能改变Jung训练的3% THD+N增强。"""
    t, clean, omega, config = source_record
    noisy = runner.noise_records({'r': (t, clean, omega)}, 42, 1,
                                method='jung', meta=[], config=config)['r'][1]
    dc, _, _ = record_reference(t, clean, tuple(config['reference_window_s']))
    fundamental, native = fundamental_thd(t, clean, omega, dc)
    expected = fundamental * np.sqrt(np.maximum((config['jung_thdn_percent'] / 100) ** 2 - native ** 2, 0))
    measured = np.sqrt(np.mean((noisy - clean) ** 2, axis=0))
    np.testing.assert_allclose(measured, expected, rtol=0.05, atol=1e-10)


def test_hang_training_keeps_snr_schedule(source_record):
    """Hang训练仍按冻结配置在15/20 dB中选取当前记录的噪声档。"""
    t, clean, omega, config = source_record
    noisy = runner.noise_records({'r': (t, clean, omega)}, 42, 1,
                                method='hang', meta=[], config=config)['r'][1]
    levels = config['hang_train_snr_db']
    expected = levels[(1 + 0 + 42) % len(levels)]
    achieved = 10 * np.log10(np.mean(clean ** 2, axis=0) / np.mean((noisy - clean) ** 2, axis=0))
    np.testing.assert_allclose(achieved, np.full(5, expected), atol=0.25, rtol=0)
