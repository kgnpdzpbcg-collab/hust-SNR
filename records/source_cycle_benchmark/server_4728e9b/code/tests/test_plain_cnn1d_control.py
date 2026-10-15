"""Tests for residual-ablated CNN1D and the source-only protocol.

Run on the original hust-SNR environment; tests contain no target data access.
"""
import json
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from plain_cnn1d_baseline import FivePhasePlainCNN1D, PlainConvBlock


def test_raw_input_and_feature_shapes():
    m=FivePhasePlainCNN1D().eval()
    with torch.no_grad():
        logits,features=m(torch.randn(3,5,4096),True)
    assert logits.shape==(3,6)
    assert features.shape==(3,128)


def test_no_residual_shortcut_modules_or_state():
    m=FivePhasePlainCNN1D()
    assert sum(isinstance(x,PlainConvBlock) for x in m.modules())==6
    assert not any("shortcut" in name.lower() or "skip" in name.lower()
                   for name in m.state_dict())
    assert not any(type(x).__name__=="ResidualBlock" for x in m.modules())


def test_stages_and_downsampling_align_resnet():
    m=FivePhasePlainCNN1D().eval()
    with torch.no_grad():
        norm=m.normalize(torch.randn(2,5,4096))
        a=m.stem(norm)
        b=m.layers(a)
    assert norm.shape==(2,5,128)
    assert a.shape==(2,32,128)
    assert b.shape==(2,128,32)
    assert [m.layers[i].body[0].stride[0] for i in (0,1,2,3,4,5)]==[1,1,2,1,2,1]


def test_backprop_reaches_early_stem():
    m=FivePhasePlainCNN1D().train()
    output=m(torch.randn(4,5,4096))
    output.square().mean().backward()
    assert m.stem[0].weight.grad is not None
    assert torch.isfinite(m.stem[0].weight.grad).all()
    assert float(m.stem[0].weight.grad.abs().sum())>0


def test_invalid_shapes_are_rejected():
    m=FivePhasePlainCNN1D()
    with pytest.raises(ValueError):m(torch.randn(3,4,4096))
    with pytest.raises(ValueError):m(torch.randn(3,5,128))


def test_frozen_config_scope_and_preserved_methods():
    config=json.loads((ROOT/"configs/plain_cnn1d_control_v1.json").read_text())
    assert config["method"]=="cnn1d_plain"
    assert config["seeds"]==[42,43,44]
    assert config["source_val_snr_db"]==[0,5,10,20,30]
    assert config["target_training"] is False
    assert config["final_holdout_evaluated"] is False
    assert config["target_checkpoint_selection"] is False


def test_selection_tie_breaking_source_only():
    from v1_noise_consistency import score_checkpoints
    rows=[dict(epoch=1,clean_f1=1.,noise_mean_f1=.8),
          dict(epoch=2,clean_f1=.98,noise_mean_f1=.99),
          dict(epoch=3,clean_f1=.996,noise_mean_f1=.85)]
    assert score_checkpoints(rows)["epoch"]==3


def test_uses_existing_paired_clean_noisy_cross_entropy():
    from classical_baseline_experiment import _resnet_loss
    m=FivePhasePlainCNN1D()
    clean=torch.randn(3,5,4096)
    noisy=clean+.02*torch.randn_like(clean)
    labels=torch.tensor([0,1,2])
    result=_resnet_loss(m,clean,noisy,labels)
    assert torch.isfinite(result)
    result.backward()
    assert m.classifier.weight.grad is not None
