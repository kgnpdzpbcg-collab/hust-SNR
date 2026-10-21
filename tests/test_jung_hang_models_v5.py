"""Synthetic CPU tests of the V5 paper-critical representations and training math."""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from jung_hang_models_v5 import (
    FAULT_PHASES, forward_two_cycle_indices, jung_window, hang_trajectory,
    jung_thdn_augment, _fundamental_thd_rms, _fault_quiet_mask,
)
from compare_jung_hang_v5 import build_inputs, get_model
from jung_hang_optim_v5 import AdaMod


def five_phase_record(f0=20.0, duration=1.0, fs=10000):
    time = np.arange(0, duration + 1 / fs / 2, 1 / fs)
    phase = 2 * np.pi * np.arange(5) / 5
    current = np.stack([np.sin(2 * np.pi * f0 * time - a) for a in phase], axis=1)
    omega = np.full_like(time, 2 * np.pi * f0)
    return time, current, omega


def config():
    return {"hang_points":128,"hang_image_size":40,
            "hang_phase_angles_deg":[0,72,144,216,288],"hang_quiet_ratio":0.03}


def test_forward_pair_strict_one_cycle_stride():
    rows = [dict(record_id="A", label=1, start=float(i), end=float(i + 1))
            for i in (0, 1, 2, 5, 6)]
    pairs = forward_two_cycle_indices(rows)
    assert [p[0] for p in pairs] == [0, 1, 3]
    assert [(p[1],p[2]) for p in pairs] == [(0.,2.),(1.,3.),(5.,7.)]
    assert len({(p[1],p[2]) for p in pairs}) == len(pairs)


def test_pair_not_cross_source_or_label():
    rows = [
        dict(record_id="A",label=1,start=0.,end=1.),
        dict(record_id="B",label=1,start=1.,end=2.),
        dict(record_id="A",label=2,start=1.,end=2.),
    ]
    with pytest.raises(ValueError, match="no valid"):
        forward_two_cycle_indices(rows)


def test_jung_input_and_label_anchor():
    time, currents, omega = five_phase_record(f0=20)
    rows = [dict(record_id="A",label=1,start=i / 20,end=(i + 1) / 20)
            for i in range(4)]
    data = {"A":(time,currents,omega)}
    x, anchors = build_inputs("jung",rows,data,config())
    assert x.shape == (3,5,256)
    assert anchors.tolist() == [0,1,2]
    assert abs(x).max() <= 1.00001
    assert np.isfinite(x).all()


def test_hang_uses_single_normalized_trajectory():
    t,c,w = five_phase_record()
    x = hang_trajectory(t,c,0,1,points=128,size=40)
    assert x.shape == (1,40,40)
    assert x.dtype == np.float32
    assert x.sum() > 0
    assert np.array_equal(x, x.astype(bool).astype(np.float32))
    # A phase-wise common amplitude should not change the normalized trajectory.
    y = hang_trajectory(t,4*c,0,1,points=128,size=40)
    np.testing.assert_array_equal(x,y)


def test_hang_mask_does_not_bridge_zero_current():
    t,c,_ = five_phase_record()
    # synthetic entire-array zero under a window has no bogus arc.
    x = hang_trajectory(t,0*c,0,1,points=128,size=40)
    assert x.sum() == 0


def test_thdn_awgn_calibrated_from_healthy_reference():
    t,c,w = five_phase_record(f0=20.0)
    fundamental, thd = _fundamental_thd_rms(t,c,w)
    assert np.all(fundamental > 0.6)
    assert np.all(thd < 0.025)
    rng = np.random.default_rng(123)
    noisy = jung_thdn_augment(t,c,w,10.0,rng,metadata=(),limiter="none")
    measured = np.sqrt(np.mean((noisy-c)**2,axis=0))
    expected = fundamental * np.sqrt(np.maximum(0.1**2-thd**2,0))
    np.testing.assert_allclose(measured,expected,rtol=1e-10,atol=1e-10)


def test_fault_specific_quiet_mask_and_zero_noise_intervals():
    t,c,w = five_phase_record(f0=20.0)
    faulty = c.copy()
    in_fault = (t>=0.5)&(t<0.95)
    faulty[in_fault,0]=0
    rows=[dict(record_id="A", label=3, start=0.5,end=0.95)]
    mask = _fault_quiet_mask(t,faulty,w,rows)
    assert np.all(mask[in_fault,1:])
    assert not mask[in_fault,0].any()
    out = jung_thdn_augment(t,faulty,w,10,np.random.default_rng(5),
                             metadata=rows,limiter="fault_quiet")
    np.testing.assert_allclose(out[in_fault,0],faulty[in_fault,0])


def test_adamod_bounds_and_gradients():
    p=torch.nn.Parameter(torch.tensor([2.0, -2.0]))
    optim=AdaMod([p],lr=1e-2)
    for _ in range(5):
        loss=p.square().sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert p.detach().abs().max().item()<2
    state=optim.state[p]
    assert state["step"]==5
    assert torch.isfinite(state["s"]).all()


@pytest.mark.parametrize("method",["jung","hang"])
def test_model_cpu_backprop(method):
    torch.manual_seed(3)
    model=get_model(method,config())
    x=torch.randn((2,5,256) if method=="jung" else (2,1,40,40))
    y=model(x)
    assert y.shape==(2,6)
    y.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
