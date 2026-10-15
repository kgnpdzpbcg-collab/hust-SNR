"""验证七模型协议、原损失分派及六种子接口，不读取真实实测数据。"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import source_cycle_benchmark as benchmark

torch.set_num_threads(2)


def test_requested_seven_models_six_seeds_and_data_version():
    config = benchmark.configuration()
    assert config["methods"] == ["resnet1d", "cnn1d_plain", "xu_full", "resnet_lp3", "resnet_lp4", "b0", "b1"]
    assert config["seeds"] == [42, 43, 44, 2024, 2025, 2026]
    assert (config["source_train_cycles"], config["source_val_cycles"], config["raw_points"]) == (962, 221, 4096)
    assert config["epochs"] == 50 and config["batch_size"] == 32
    assert config["target_conditions"] == ["clean", 10, 15, 20, 25, 30, 35, 40, 5, 0, -5, -10]
    assert config["target_training"] is False and config["final_holdout_evaluated"] is False


@pytest.mark.parametrize("method", benchmark.METHODS)
def test_original_models_forward_and_loss_gradients(method):
    """检查实际原类的分类输出和关键参数梯度，避免仅替换模型名字。"""
    torch.manual_seed(2024)
    model = benchmark.build_model(method)
    clean = torch.randn(2, 5, 4096)
    noisy = clean + 0.15 * torch.randn_like(clean)
    labels = torch.tensor([1, 5])
    loss, terms = benchmark.objective(method, model, clean, noisy, labels, benchmark.configuration())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    if method == "xu_full":
        assert set(terms) == {"ce", "recon", "dml", "intra", "inter"}
        assert float(loss.detach()) == pytest.approx(float(terms["ce"] + terms["recon"] + terms["dml"]), rel=1e-6)
        assert model.network.decoder[1][0].weight.grad is not None
    elif method == "b0":
        assert float(loss.detach()) == pytest.approx(float(terms["ce"] + 0.2 * terms["rec"]), rel=1e-6)
    elif method == "b1":
        assert float(loss.detach()) == pytest.approx(float(terms["ce"] + terms["restore"]), rel=1e-6)
    predicted, ce = benchmark.predict(model, clean.numpy(), torch.device("cpu"), labels.numpy())
    assert predicted.shape == (2,) and np.isfinite(ce)


def test_common_selector_uses_source_ce_only_after_f1_ties():
    history = [
        {"epoch": 1, "clean_f1": 1.0, "noise_mean_f1": 1.0, "noise_mean_ce": 0.5},
        {"epoch": 2, "clean_f1": 1.0, "noise_mean_f1": 1.0, "noise_mean_ce": 0.01},
        {"epoch": 3, "clean_f1": 1.0, "noise_mean_f1": 0.99, "noise_mean_ce": 0.001},
    ]
    assert benchmark.select_source_checkpoint(history)["epoch"] == 2


@pytest.mark.parametrize("seed", benchmark.SEEDS)
def test_cli_routes_every_requested_seed_without_training(monkeypatch, tmp_path, seed):
    calls = []
    monkeypatch.setattr(benchmark, "train", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["benchmark", "train", "--methods", "resnet1d", "--seeds", str(seed), "--out", str(tmp_path)])
    benchmark.main()
    assert calls[0][1] == seed


def test_foreign_data_version_is_rejected_before_loading_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "source_provenance", lambda root: {"manifest_sha256": "old-split"})
    with pytest.raises(ValueError, match="划分指纹"):
        benchmark.checked_source(tmp_path, benchmark.configuration())


def test_snapshot_is_immutable_and_includes_bn_buffers():
    model = benchmark.build_model("resnet1d")
    state = benchmark.snapshot(model)
    name = next(k for k in state if k.endswith("running_mean"))
    original = state[name].clone()
    model.state_dict()[name].add_(1)
    assert torch.equal(state[name], original)


def test_summary_rejects_different_target_cycle_order(tmp_path):
    for condition, truth in (("clean", [0]), ("snr10", [1])):
        benchmark.write(tmp_path / f"evaluations/resnet1d/seed42/target_dev_{condition}.json", {
            "method": "resnet1d", "seed": 42, "protocol": benchmark.PROTOCOL,
            "condition": condition, "record_families": 88, "windows": 2474,
            "truth": truth, "source_split": {"protocol": "same"}, "final_holdout_evaluated": False,
            "metrics": {"macro_f1": 1, "accuracy": 1, "missed_fault_as_normal": 0,
                        "normal_false_alarms": 0, "per_class_f1": [1] * 6, "per_class_recall": [1] * 6},
        })
    with pytest.raises(ValueError, match="不同数据划分或不同目标队列"):
        benchmark.summarize(tmp_path)
