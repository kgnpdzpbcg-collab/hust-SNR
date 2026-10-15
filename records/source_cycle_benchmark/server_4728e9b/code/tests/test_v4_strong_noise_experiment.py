"""核查真实强噪声训练路径、固定验证MAT及旧协议不被全局修改。"""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v2_experiment as v2
import v4_strong_noise_experiment as strong


def synthetic():
    """与真实仿真相同的10000行、五相电流和角速度格式。"""
    t = np.arange(10000) / 10000
    current = np.stack([np.sin(2*np.pi*20*t + phase) for phase in np.arange(5)*.2], axis=1)
    raw = {"sample": (t, current, np.full(10000, 2*np.pi*20))}
    return raw, [{"record_id": "sample", "label": 1, "start": .6, "end": .65}]


def test_legacy_noise_is_numerically_unchanged():
    """展开旧实现的随机流，确认默认调用仍生成逐数值相同的周期。"""
    raw, metadata = synthetic()
    rng = np.random.default_rng(42 + 15000 + 1)
    snr = {"sample": int(rng.choice(v2.NOISE_LEVELS))}
    noisy = v2.calibrated_noise(raw, snr, lambda _: np.random.default_rng(42 + 15000 + 1 + 997))
    assert np.array_equal(v2.raw_cycles(noisy, metadata), v2.fresh_noisy_source(raw, metadata, 42, 1))
    assert v2.NOISE_LEVELS == (10, 15, 20, 25, 30, 35, 40)
    assert v2.VAL_NOISE_LEVELS == (10, 20, 30)


def test_strong_noise_enters_training_and_changes_each_epoch():
    """本轮九档均进入计划，且相同种子可复现；两轮波形有变化。"""
    raw, metadata = synthetic()
    levels = strong.config()["training_noise_levels_db"]
    schedules = [v2.training_noise_schedule([f"record{i}" for i in range(34)], 42, e, levels) for e in range(1, 51)]
    assert set(v for r in schedules for v in r.values()) == set(levels)
    a = v2.fresh_noisy_source(raw, metadata, 42, 1, noise_levels=levels)
    assert np.array_equal(a, v2.fresh_noisy_source(raw, metadata, 42, 1, noise_levels=levels))
    assert not np.array_equal(a, v2.fresh_noisy_source(raw, metadata, 42, 2, noise_levels=levels))


def test_materialized_validation_matches_fixed_rng_and_requires_files(tmp_path):
    """已保存0/5 dB与原固定Val随机流相同，读取失败必须报错而非重抽。"""
    raw, metadata = synthetic()
    t, current, omega = raw["sample"]
    for level in (0, 5):
        digest = hashlib.sha256(f"HUST-V1:val:{level}:sample".encode()).digest()
        seed = int.from_bytes(digest[:8], "little")
        noisy = v2.calibrated_noise(raw, {"sample": level}, lambda _: np.random.default_rng(seed))["sample"][1]
        directory = tmp_path / f"SNR_{level}"
        directory.mkdir()
        savemat(directory / "sample.mat", {"I_We": np.column_stack([noisy, omega])})
    saved = v2.fixed_noisy_validation(raw, metadata, noise_levels=(0, 5), materialized_root=tmp_path)
    expected = v2.fixed_noisy_validation(raw, metadata, noise_levels=(0, 5))
    assert all(np.array_equal(saved[k], expected[k]) for k in (0, 5))
    (tmp_path / "SNR_0/sample.mat").unlink()
    with pytest.raises(OSError):
        v2.fixed_noisy_validation(raw, metadata, noise_levels=(0,), materialized_root=tmp_path)


def test_old_checkpoint_cannot_be_reported_as_strong_retraining():
    """独立入口必须拒绝只有原训练协议的权重。"""
    protocol = strong.config()
    with pytest.raises(ValueError, match="强噪声重训"):
        strong.check_checkpoint({"train_protocol": "V2_M1_paired_source_CE"}, protocol)
    strong.check_checkpoint({"train_protocol": strong.PROTOCOL,
                             "train_noise_levels_db": protocol["training_noise_levels_db"],
                             "val_noise_levels_db": protocol["validation_noise_levels_db"]}, protocol)
