"""Fast offline integrity tests. Official result requires server source data."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from classical_baseline_models import (PHASES, RAW_POINTS, FEATURE_VERSION,
    engineered_features, md_normalize_numpy, FivePhaseResNet1D, ResidualBlock)
from classical_baseline_experiment import (_selection, classical_candidates,
    _resnet_loss, config, _classical_training_arrays, _write, _csv)


def test_spec_is_frozen():
    spec=config()
    assert spec["seeds"]==[42,43,44]
    assert spec["target_dev_recordings"]==88
    assert spec["target_dev_cycles_per_condition"]==2474
    assert spec["classical"]["feature_dimension"]==120
    assert spec["classical"]["feature_version"]==FEATURE_VERSION
    assert spec["target_used_for_training_or_selection"] is False
    assert spec["final_holdout_evaluated"] is False


def test_md_low_current_threshold_exact():
    raw=np.zeros((2,PHASES,RAW_POINTS),np.float32)
    t=np.arange(RAW_POINTS,dtype=np.float32)
    raw[0,0]=.1*np.sin(2*np.pi*t/RAW_POINTS)
    raw[1,0]=2.0*np.sin(2*np.pi*t/RAW_POINTS)
    y=md_normalize_numpy(raw)
    assert np.max(np.abs(y[0,0]))==pytest.approx(.001,abs=1e-7)
    assert np.max(np.abs(y[1,0]))==pytest.approx(1.,abs=1e-7)
    assert np.isfinite(y).all()


def test_features_deterministic_and_finite_with_missing_phase():
    raw=np.zeros((4,5,4096),np.float32)
    theta=np.linspace(0,2*np.pi,4096,endpoint=False,dtype=np.float32)
    for p in range(5):raw[:,p,:]=np.sin(theta+p*2*np.pi/5)
    raw[0,3]=0
    a=engineered_features(raw,batch_size=1)
    b=engineered_features(raw,batch_size=4)
    assert a.shape==(4,120) and np.array_equal(a,b)
    assert np.isfinite(a).all()
    assert not np.array_equal(a[0],a[1])


def test_harmonic_amplitude_identifies_order():
    theta=np.linspace(0,2*np.pi,4096,endpoint=False,dtype=np.float32)
    raw=np.zeros((1,5,4096),np.float32)
    raw[0,0]=np.sin(3*theta)
    ft=engineered_features(raw)[0].reshape(5,24)
    amps=ft[0,8:16]
    assert amps.argmax()==2  # 3rd harmonic, indexing from k=1
    assert amps[2]>amps[0]*100


def test_normalizer_rejects_shape_nonfinite():
    with pytest.raises(ValueError):engineered_features(np.zeros((2,5,128)))
    with pytest.raises(ValueError):md_normalize_numpy(np.full((1,5,4096),np.nan))


def test_resnet_is_genuine_deep_residual_backbone():
    m=FivePhaseResNet1D()
    blocks=[z for z in m.layers if isinstance(z,ResidualBlock)]
    assert len(blocks)==6
    assert sum(p.numel() for p in m.parameters())>25620
    x=torch.randn(2,5,4096)
    logits,features=m(x,return_features=True)
    assert logits.shape==(2,6) and features.shape==(2,128)
    logits.sum().backward()
    assert m.stem[0].weight.grad is not None
    assert torch.isfinite(m.stem[0].weight.grad).all()


def test_resnet_paired_loss_backward():
    m=FivePhaseResNet1D()
    x=torch.randn(3,5,4096)
    l=_resnet_loss(m,x,x+.1*torch.randn_like(x),torch.tensor([0,1,2]))
    l.backward()
    assert torch.isfinite(l) and m.classifier.weight.grad.abs().sum()>0


def test_only_source_selection_clean_guard():
    rows=[dict(candidate_index=i,clean_f1=cl,noise_mean_f1=no)
          for i,(cl,no) in enumerate([(1.0,.60),(.97,.99),(.995,.80),(.995,.70)],start=1)]
    assert _selection(rows)["candidate_index"]==3
    # Higher noisy F1 may win if it has clean within 1pp of best.
    rows[3]["noise_mean_f1"]=.89
    assert _selection(rows)["candidate_index"]==4


def test_svm_rf_candidates_and_train_only_scaler():
    x=np.random.default_rng(42).normal(size=(36,120)).astype(np.float32)
    y=np.arange(36)%6
    for name in ("svm_rbf","random_forest"):
        choices=list(classical_candidates(name,42))
        assert len(choices)>=2
        params,model=choices[0]
        model.fit(x,y)
        assert model.predict(x[:6]).shape==(6,)
        if name=="svm_rbf":
            scaler=model.named_steps["scaler"]
            np.testing.assert_allclose(scaler.mean_,x.mean(axis=0),rtol=1e-4,atol=1e-5)


def test_source_augmented_views_only_from_source(monkeypatch):
    import classical_baseline_experiment as exp
    raw=np.zeros((6,5,4096),np.float32)
    labels=np.arange(6,dtype=np.int64)
    calls=[]
    def fake_fresh(tr,tm,seed,epoch):
        calls.append((seed,epoch))
        return raw+(epoch*.001)
    monkeypatch.setattr(exp.data,"fresh_noisy_source",fake_fresh)
    features,y=exp._classical_training_arrays(42,{},[],labels,raw)
    assert features.shape==(48,120) and len(y)==48
    assert calls==[(42,i) for i in range(1,8)]
    assert np.array_equal(y,np.tile(labels,8))


def test_no_overwrite_of_results(tmp_path):
    p=tmp_path/"safe.json"
    _write(p,{"source_only":True})
    assert json.loads(p.read_text())["source_only"]
    with pytest.raises(FileExistsError):_write(p,{"source_only":False})
    c=tmp_path/"r.csv"
    _csv(c,[{"method":"svm_rbf","macro_f1":.8}])
    with pytest.raises(FileExistsError):_csv(c,[{"method":"random_forest","macro_f1":.5}])
