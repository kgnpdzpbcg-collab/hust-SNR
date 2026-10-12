"""CPU-only deterministic unit tests for noise consistency V1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import torch
from v1_noise_consistency import CNN, consistency_loss, train_objective, metrics, score_checkpoints


def test_feature_size_and_state_dict_compatibility():
    torch.manual_seed(1)
    model = CNN()
    logits, feats = model(torch.randn(4, 5, 128), return_features=True)
    assert logits.shape == (4, 6) and feats.shape == (4, 50)
    assert sum(p.numel() for p in model.parameters()) == 25620
    assert "classifier.1.weight" in model.state_dict()


def test_consistency_same_view_zero_and_grad():
    m = CNN()
    x = torch.randn(6, 5, 128)
    clean, noisy = x, x + 0.3 * torch.randn_like(x)
    assert abs(float(consistency_loss(m(clean, True)[1], m(clean, True)[1]).detach())) < 1e-6
    for method in ("baseline", "augmentation", "consistency"):
        m.zero_grad(set_to_none=True)
        loss, stats = train_objective(m, clean, torch.arange(6), None if method == "baseline" else noisy, method, .1)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with pytest.raises(ValueError):
        train_objective(m, clean, torch.arange(6), noisy, "baseline")


def test_consistency_objective_exact_difference_in_eval_mode():
    m = CNN().eval()
    x = torch.randn(6, 5, 128)
    xx = x + 0.1 * torch.randn_like(x)
    y = torch.arange(6)
    a, _ = train_objective(m, x, y, xx, "augmentation", .1)
    b, aux = train_objective(m, x, y, xx, "consistency", .1)
    assert float(b.detach()) == pytest.approx(float(a.detach()) + 0.1 * aux["cons"], abs=1e-6)


def test_checkpoint_selection_clean_guard_then_noisy():
    h = [dict(epoch=1, clean_f1=1., noise_mean_f1=.4),
         dict(epoch=2, clean_f1=.99, noise_mean_f1=.7),
         dict(epoch=3, clean_f1=.98, noise_mean_f1=.9)]
    assert score_checkpoints(h)["epoch"] == 2


def test_normal_false_alarm_not_mixed_with_false_negative():
    # true normal[0,0] => second normal predicted as fault (FAR=50%)
    y = [0,0,1,1,2,3,4,5]
    pred = [0,2,0,1,2,3,4,5]
    m=metrics(y,pred)
    assert m["normal_false_alarm_rate"] == .5
    assert m["normal_false_alarms"] == 1
    assert m["missed_fault_as_normal"] == 1
    assert len(m["confusion_matrix"]) == 6
    assert len(m["confusion_matrix"]) == 6