"""Frozen 9-arm architecture and source-only experiment protocol tests.

Run from repository root: python -m pytest -q tests/test_hust_arch_suite.py
No real MAT, target labels, or target Final-Holdout are opened by this suite.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hust_arch_suite_models import (
    METHODS, HUSTArchitectureCandidate, WideAEncoder, MultiScaleStem,
    EqualParamDoubleConv, FixedLowpass4, paired_ce, parameter_count,
)


def test_registry_and_config_exact_frozen_scope():
    c = json.loads((ROOT / "configs/hust_arch_suite_v1.json").read_text())
    assert len(METHODS) == len(set(METHODS)) == 9
    assert list(METHODS) == c["methods"]
    assert c["seeds"] == [42, 43, 44]
    assert c["source_train_snr_db"] == [10, 15, 20, 25, 30, 35, 40]
    assert c["source_val_snr_db"] == [0, 5, 10, 20, 30]
    assert c["epochs"] == 50 and c["batch_size"] == 32
    assert not c["target_training"] and not c["target_checkpoint_selection"]
    assert not c["final_holdout_evaluated"]


@pytest.mark.parametrize("name", METHODS)
def test_each_architecture_output_and_features(name):
    torch.manual_seed(17)
    model = HUSTArchitectureCandidate(name).eval()
    with torch.no_grad():
        x = torch.randn(2, 5, 4096)
        logits, features = model(x, return_features=True)
    assert logits.shape == (2, 6)
    assert features.shape == (2, 128 if name.startswith("resnet_") else 704)
    assert torch.isfinite(logits).all() and torch.isfinite(features).all()


def test_wide_stem_temporal_sizes_match_a_prime():
    x = torch.randn(2, 5, 128)
    m = WideAEncoder(multiscale=False, extra="none").eval()
    with torch.no_grad():
        assert m.stem(x).shape == (2, 32, 56)
        assert m.stage2(m.stem(x)).shape == (2, 64, 26)
        assert m.stage3(m.stage2(m.stem(x))).shape == (2, 64, 11)


def test_ms_all_branches_align_and_gradients_reach_every_branch():
    m = MultiScaleStem()
    x = torch.randn(2, 5, 128)
    out = m(x)
    assert out.shape == (2, 32, 113)
    out.square().mean().backward()
    assert [a.kernel_size[0] for a in m.branches] == [7, 15, 31]
    assert [a.out_channels for a in m.branches] == [8, 12, 12]
    assert all(z.weight.grad is not None and z.weight.grad.abs().sum().item() > 0
               for z in m.branches)


@pytest.mark.parametrize("is_ms", [False, True])
def test_equal_parameter_plain_residual_and_equal_initial_weights(is_ms):
    # The only intervention is a parameter-free identity shortcut.
    torch.manual_seed(33)
    plain = WideAEncoder(multiscale=is_ms, extra="plain")
    torch.manual_seed(33)
    res = WideAEncoder(multiscale=is_ms, extra="residual")
    assert parameter_count(plain) == parameter_count(res)
    pa, rb = plain.state_dict(), res.state_dict()
    assert pa.keys() == rb.keys()
    assert all(torch.equal(pa[k], rb[k]) for k in pa)
    assert not plain.extra.residual and res.extra.residual
    assert plain.extra.body is not res.extra.body
    assert all(not n.startswith("shortcut") for n in pa)


def test_plain_residual_math_same_body_no_trainable_skip():
    torch.manual_seed(123)
    plain = EqualParamDoubleConv(False).eval()
    torch.manual_seed(123)
    res = EqualParamDoubleConv(True).eval()
    x = torch.randn(2, 64, 11)
    with torch.no_grad():
        body = plain.body(x)
        expected_plain = plain.activation(body)
        expected_residual = res.activation(body + x)
        assert torch.allclose(plain(x), expected_plain)
        assert torch.allclose(res(x), expected_residual)
        assert not torch.allclose(plain(x), res(x))


def test_lowpass4_mask_preserves_dc_through_four_and_blocks_higher():
    x = torch.arange(4096, dtype=torch.float32)
    t = 2*torch.pi*x/4096
    wave = (0.4 + 1.1*torch.cos(t) + 0.6*torch.sin(4*t) +
            2.2*torch.cos(5*t)).expand(2, 5, -1).contiguous()
    filtered = FixedLowpass4()(wave)
    desired = (0.4 + 1.1*torch.cos(t) + 0.6*torch.sin(4*t))
    assert torch.allclose(filtered[0,0], desired, atol=2e-5, rtol=2e-5)


def test_lp3_lp4_are_distinct_when_fourth_harmonic_exists():
    x = torch.randn(2, 5, 4096)
    from v2_paifilter import HarmonicShaping
    a = HarmonicShaping("lowpass3")(x)
    b = FixedLowpass4()(x)
    assert torch.max((a-b).abs()) > 0


def test_no_fft_bypass_of_frontend_in_resnet_composites():
    for name in ("resnet_lp3", "resnet_lp4", "resnet_nahsf"):
        model = HUSTArchitectureCandidate(name)
        assert model.normalize.__class__.__name__ == "MDNormalization"
        assert model.backbone.__class__.__name__ == "FivePhaseResNet1D"
        # ResNet.forward is intentionally not called; its internal normalize
        # must never be reached by the wrapper's normalized-128 pathway.
        called = []
        def prohibit(*args, **kwargs):
            called.append(True)
            raise AssertionError("ResNet double-normalization attempted")
        model.backbone.normalize.forward = prohibit
        with torch.no_grad():
            y = model.eval()(torch.randn(1,5,4096))
        assert y.shape == (1,6) and not called


def test_source_only_paired_ce_backprop_in_both_network_families():
    torch.manual_seed(7)
    x = torch.randn(3,5,4096)
    y = torch.tensor([0,3,5])
    for name in ("v5_wd_res_ms", "resnet_nahsf"):
        m = HUSTArchitectureCandidate(name).train()
        loss = paired_ce(m, x, x + .1*torch.randn_like(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        assert m.backbone.classifier[1].weight.grad is not None if name.startswith("v5_") else m.backbone.classifier.weight.grad is not None
        gate_weights = m.shaping.gate[-1].weight.grad
        assert gate_weights is not None and torch.isfinite(gate_weights).all()


def test_invalid_inputs_rejected():
    m = HUSTArchitectureCandidate("v5_w")
    with pytest.raises(ValueError):
        m(torch.randn(2,4,4096))
    with pytest.raises(ValueError):
        m(torch.randn(2,5,128))


def test_source_selection_and_pure_result_writer(tmp_path):
    from v1_noise_consistency import score_checkpoints
    from hust_arch_suite_experiment import write_json, write_csv, configuration
    assert configuration()["protocol"].endswith("source-only")
    candidates=[dict(epoch=1,clean_f1=1.0,noise_mean_f1=.80),
                dict(epoch=2,clean_f1=.980,noise_mean_f1=.99),
                dict(epoch=3,clean_f1=.996,noise_mean_f1=.85)]
    assert score_checkpoints(candidates)["epoch"] == 3
    out=tmp_path/"test.json"
    write_json(out, {"a":1})
    with pytest.raises(FileExistsError):write_json(out,{"a":2})
    csv_path=tmp_path/"results.csv"
    write_csv(csv_path,[{"a":2,"b":3}])
    with pytest.raises(FileExistsError):write_csv(csv_path,[{"a":4,"b":5}])


def test_frontend_gate_is_trainable_and_source_only():
    m=HUSTArchitectureCandidate("v5_w")
    x=torch.randn(2,5,4096)
    z=m.shaping(x)
    assert z.shape==x.shape
    assert m.shaping.gate[0].in_features==3
    assert m.shaping.gate[0].out_features==8
    assert m.shaping.gate[-1].out_features==1


def test_architecture_family_param_ordering():
    m={k:parameter_count(HUSTArchitectureCandidate(k)) for k in METHODS}
    assert m["v5_wd_res"]==m["v5_wd_plain"]
    assert m["v5_wd_res_ms"]==m["v5_wd_plain_ms"]
    assert m["v5_wd_res"]>m["v5_w"]
    assert m["v5_wd_res_ms"]>m["v5_w_ms"]
    assert m["resnet_lp3"]==m["resnet_lp4"]
    assert m["resnet_nahsf"]==m["resnet_lp3"]+41


def test_resnet_initial_backbone_parity_across_all_three_spectral_arms():
    # Same torch seed must initialize SAME reference ResNet backbone for
    # LP3, LP4 and NA-HSF. Only spectral shaping differs.
    states=[]
    for method in ("resnet_lp3", "resnet_lp4", "resnet_nahsf"):
        torch.manual_seed(44)
        m=HUSTArchitectureCandidate(method)
        states.append({key:v.detach().clone() for key,v in m.backbone.state_dict().items()})
    assert states[0].keys()==states[1].keys()==states[2].keys()
    assert all(torch.equal(states[0][k],states[i][k])
               for k in states[0] for i in (1,2))


def test_resnet_lp3_and_lp4_clean_forward_not_identical():
    torch.manual_seed(101)
    lp3=HUSTArchitectureCandidate("resnet_lp3").eval()
    torch.manual_seed(101)
    lp4=HUSTArchitectureCandidate("resnet_lp4").eval()
    idx=torch.arange(4096,dtype=torch.float32)
    t=2*torch.pi*idx/4096
    x=(1.5*torch.cos(t)+torch.sin(4*t)+.4*torch.cos(6*t))[None,None,:].expand(2,5,-1).contiguous()
    with torch.no_grad():
        y3=lp3(x);y4=lp4(x)
    assert y3.shape==y4.shape==(2,6)
    assert not torch.allclose(y3,y4,rtol=1e-5,atol=1e-5)
