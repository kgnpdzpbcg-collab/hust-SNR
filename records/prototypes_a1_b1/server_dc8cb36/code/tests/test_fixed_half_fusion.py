"""验证固定融合的频域数学定义、参数、初始化和协议隔离。"""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fixed_half_fusion import FixedHalfCNN, FixedHalfFusion
from v5_nahsf import NAHSFCNN
from v4_hpasf import HPASFCNN
import fixed_half_experiment as experiment
import four_models_expanded_val as common


def test_fixed_mask_matches_midpoint_and_phase():
    """LP3/LP4算术均值与固定掩码等价；第四阶复系数保持相位并减半。"""
    raw = torch.randn(3, 5, 4096, dtype=torch.float64)
    module = FixedHalfFusion()
    actual = module(raw)
    midpoint = .5*HPASFCNN("lowpass3").shaping(raw) + .5*HPASFCNN("lowpass4").shaping(raw)
    torch.testing.assert_close(actual, midpoint, atol=1e-14, rtol=1e-14)
    before, after = (torch.fft.rfft(x, norm="ortho") for x in (raw, actual))
    torch.testing.assert_close(after[:, :, :4], before[:, :, :4], atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(after[:, :, 4], .5*before[:, :, 4], atol=1e-14, rtol=1e-14)
    assert after[:, :, 5:].abs().max() < 1e-14


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_backbone_and_initial_v5_logits_exact(seed):
    """固定模型和初始V5计算完全相同，但固定模型没有描述符或MLP。"""
    torch.manual_seed(seed)
    fixed = FixedHalfCNN().eval()
    torch.manual_seed(seed)
    v5 = NAHSFCNN().eval()
    assert all(torch.equal(a, b) for a, b in zip(fixed.backbone.parameters(), v5.backbone.parameters()))
    raw = torch.randn(4, 5, 4096)
    assert torch.equal(fixed.shaping(raw), v5.shaping(raw))
    assert torch.equal(fixed(raw), v5(raw))
    assert common.parameter_counts(fixed) == {"cnn": 25620, "spectral": 0, "total": 25620}
    assert not hasattr(fixed.shaping, "gate") and not hasattr(fixed.shaping, "descriptors")
    assert not list(fixed.shaping.parameters()) and not list(fixed.shaping.buffers())


def test_protocol_and_checkpoint_isolation():
    """与V5训练/选模相同；方法及版本不同，禁止混用权重或改变训练噪声。"""
    p, ref = experiment.config(), common.config()
    assert all(p[k] == ref[k] for k in experiment.FAIR_KEYS)
    assert "fixed_half" not in common.METHODS
    checkpoint = dict(method="fixed_half", seed=42, code_family=experiment.VERSION, source_only=True,
                      raw_points=4096, train_noise_levels_db=p["training_noise_levels_db"],
                      val_noise_levels_db=p["validation_noise_levels_db"], protocol=p["protocol"])
    common.check_checkpoint(checkpoint, "fixed_half", 42, protocol=p, version=experiment.VERSION)
    with pytest.raises(ValueError):
        common.check_checkpoint(checkpoint, "nahsf", 42)
    checkpoint["train_noise_levels_db"] = [0, 5, *p["training_noise_levels_db"]]
    with pytest.raises(ValueError):
        common.check_checkpoint(checkpoint, "fixed_half", 42, protocol=p, version=experiment.VERSION)


def test_reject_incomplete_or_nonfinite_cycle():
    """FFT滤波只接受完整五相周期，不把任意长度窗口当作谐波周期。"""
    for raw in (torch.zeros(1, 5, 128), torch.full((1, 5, 4096), float("nan"))):
        with pytest.raises(ValueError):
            FixedHalfFusion()(raw)
