"""追加种子必须保留原模型、CNN初值、训练/验证协议和旧结果来源。"""
from pathlib import Path
import sys
import statistics
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import five_models_extra_seeds as experiment
import four_models_expanded_val as common
from v2_paifilter import OriginalCNN


def test_only_extra_seeds_not_extra_training_noise():
    """不换数据切分，不把新增验证0/5 dB混入训练，保留原三个种子。"""
    p, old = experiment.config(), common.config()
    assert p["seeds"] == [2024, 2025, 2026] and p["combined_seeds"] == [42, 43, 44, 2024, 2025, 2026]
    assert all(p[k] == old[k] for k in experiment.FAIR_KEYS)
    assert p["training_noise_levels_db"] == [10, 15, 20, 25, 30, 35, 40]


@pytest.mark.parametrize("seed", experiment.NEW_SEEDS)
def test_five_original_backbones_pair_at_new_seeds(seed):
    """五种模型的CNN初值必须一致，V5之外不引入频域学习参数。"""
    torch.manual_seed(seed)
    backbone = OriginalCNN()
    for method in experiment.METHODS:
        torch.manual_seed(seed)
        model = experiment.make_model(method)
        assert all(torch.equal(a, b) for a, b in zip(model.backbone.parameters(), backbone.parameters()))
        assert common.parameter_counts(model) == {"cnn": 25620, "spectral": 41 if method == "nahsf" else 0,
                                                   "total": 25661 if method == "nahsf" else 25620}


def test_six_seed_statistics_use_sample_std_and_keep_outlier():
    """六种子统计必须包含异常低成绩，标准差不能误用总体标准差。"""
    values = [95., 96., 40., 98., 99., 100.]
    result = experiment.statistics(values)
    assert result["values"] == values and result["mean"] == pytest.approx(statistics.mean(values))
    assert result["std"] == pytest.approx(statistics.stdev(values))
    with pytest.raises(ValueError):
        experiment.statistics(values[:5])


def test_result_origin_and_seed_guard():
    """新种子不能伪装成旧协议，也不能将旧三档验证结果混入新统计。"""
    p = experiment.config()
    result = dict(method="nahsf", seed=2024, condition="snr0", protocol=p["protocol"], record_families=88,
                  windows=2474, target_used_for_training_or_selection=False, final_holdout_evaluated=False,
                  truth=[0]*2474, predicted=[0]*2474)
    experiment.validate_result(result, "nahsf", 2024, "snr0", p["protocol"])
    result["protocol"] = common.config()["protocol"]
    with pytest.raises(ValueError):
        experiment.validate_result(result, "nahsf", 2024, "snr0", p["protocol"])
    with pytest.raises(ValueError):
        experiment.result_root("nahsf", 99, Path("new"), Path("four"), Path("fixed"))
    assert experiment.result_root("fixed_half", 44, Path("new"), Path("four"), Path("fixed"))[0] == Path("fixed")
    assert experiment.result_root("nahsf", 42, Path("new"), Path("four"), Path("fixed"))[0] == Path("four")


def test_checkpoint_guard_isolates_extra_runs():
    """保持新旧权重元数据区分，禁止在原四模型入口冒充原种子。"""
    p = experiment.config()
    checkpoint = dict(method="nahsf", seed=2024, code_family=experiment.VERSION, source_only=True,
                      raw_points=4096, train_noise_levels_db=p["training_noise_levels_db"],
                      val_noise_levels_db=p["validation_noise_levels_db"], protocol=p["protocol"])
    common.check_checkpoint(checkpoint, "nahsf", 2024, protocol=p, version=experiment.VERSION)
    with pytest.raises(ValueError):
        common.check_checkpoint(checkpoint, "nahsf", 2024)
