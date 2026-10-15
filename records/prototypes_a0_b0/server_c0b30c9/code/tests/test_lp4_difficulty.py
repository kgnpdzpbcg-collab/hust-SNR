"""低阶对照的信息保留、预处理顺序和小模型预算核验；不读真实数据。"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lp4_difficulty_models import tensor_features, reconstruct_input128, SmallHarmonicMLP
from hust_arch_suite_models import FixedLowpass4
from v2_paifilter import MDNormalization


def test_reconstruction_preserves_original_lp4_md_input_including_low_current():
    """含低电流和全零相时，45维仍可重建原LP4＋MD网络输入。"""
    torch.manual_seed(9)
    t = torch.arange(4096) * (2 * torch.pi / 4096)
    x = (torch.sin(t)[None, None, :] * torch.tensor([1., .1, 2., .01, 0.])[None, :, None]).repeat(3, 1, 1)
    x[:, :4] += torch.randn(3, 4, 4096) * .05
    features = tensor_features(x)
    assert features.shape == (3, 45)
    expected = MDNormalization()(FixedLowpass4()(x))
    torch.testing.assert_close(reconstruct_input128(features), expected, rtol=2e-6, atol=2e-6)


def test_higher_harmonics_removed_before_normalization():
    """高阶干扰应先被LP4滤掉，不能先归一化再滤波而污染低阶幅值。"""
    t = torch.arange(4096) * (2 * torch.pi / 4096)
    clean = (torch.sin(t) + .2 * torch.cos(4 * t))[None, None].repeat(2, 5, 1)
    contaminated = clean + 3 * torch.sin(17 * t)
    torch.testing.assert_close(tensor_features(clean), tensor_features(contaminated), rtol=1e-5, atol=1e-6)


def test_phase_information_not_replaced_by_magnitude_only():
    """同幅值不同相位须得到不同的复系数表示。"""
    t = torch.arange(4096) * (2 * torch.pi / 4096)
    a = torch.sin(t)[None, None].repeat(1, 5, 1)
    b = torch.cos(t)[None, None].repeat(1, 5, 1)
    assert not torch.allclose(tensor_features(a), tensor_features(b))


def test_small_mlp_budget_and_output():
    """确保对照确为单隐藏层小MLP，输出六类且可计算梯度。"""
    model = SmallHarmonicMLP()
    assert sum(p.numel() for p in model.parameters()) == 1670
    logits = model(torch.randn(8, 45))
    assert logits.shape == (8, 6)
    torch.nn.functional.cross_entropy(logits, torch.arange(8) % 6).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
