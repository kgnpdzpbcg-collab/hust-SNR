"""检查扩大验证的四模型协议、随机数顺序、选择规则和权重隔离。"""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import four_models_expanded_val as experiment
from v2_paifilter import OriginalCNN
from v1_noise_consistency import score_checkpoints


def test_train_noise_unchanged_and_only_val_has_zero_five():
    """不得把0/5 dB意外混入本次训练。"""
    c = experiment.config()
    assert c["training_noise_levels_db"] == [10, 15, 20, 25, 30, 35, 40]
    assert c["validation_noise_levels_db"] == [0, 5, 10, 20, 30]
    assert tuple(c["methods"]) == ("identity", "lowpass3", "lowpass4", "nahsf")


def test_all_four_backbones_keep_original_initialization():
    """避免模型工厂一次构造四个对象，导致后面模型初值被随机数污染。"""
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        reference = OriginalCNN()
        for method in experiment.METHODS:
            torch.manual_seed(seed)
            model = experiment.make_model(method)
            assert all(torch.equal(a, b) for a, b in zip(model.backbone.parameters(), reference.parameters()))
            count = experiment.parameter_counts(model)
            assert count["cnn"] == 25620 and count["spectral"] == (41 if method == "nahsf" else 0)


def test_a_prime_has_exact_identity_filter():
    """A′只是不滤波，MD归一化和CNN保持一致。"""
    model = experiment.make_model("identity")
    raw = torch.randn(2, 5, 4096)
    assert torch.equal(model.shaping(raw), raw)
    assert model.normalize(raw).shape == (2, 5, 128)


def test_expanded_val_can_select_different_epoch_without_target_scores():
    """旧三档均为100时，新增0/5 dB能够区分候选轮次。"""
    history = [
        {"epoch": 1, "clean_f1": 1., "noise_mean_f1": .80, "original_three_snr_mean_f1": 1.},
        {"epoch": 2, "clean_f1": 1., "noise_mean_f1": .95, "original_three_snr_mean_f1": 1.},
    ]
    assert score_checkpoints(history)["epoch"] == 2
    assert experiment.select_original_reference(history)["epoch"] == 1


def test_checkpoint_guard_rejects_old_or_strong_training_protocol():
    """本轮入口不能评估旧三档选模、或训练含0/5 dB的其他权重。"""
    with pytest.raises(ValueError):
        experiment.check_checkpoint({"method": "nahsf", "seed": 42}, "nahsf", 42)
    c = experiment.config()
    ckpt = {"method": "nahsf", "seed": 42, "code_family": experiment.VERSION, "source_only": True,
            "raw_points": 4096, "train_noise_levels_db": c["training_noise_levels_db"],
            "val_noise_levels_db": c["validation_noise_levels_db"], "protocol": c["protocol"]}
    experiment.check_checkpoint(ckpt, "nahsf", 42)
    ckpt["train_noise_levels_db"] = [0, 5, *c["training_noise_levels_db"]]
    with pytest.raises(ValueError):
        experiment.check_checkpoint(ckpt, "nahsf", 42)
