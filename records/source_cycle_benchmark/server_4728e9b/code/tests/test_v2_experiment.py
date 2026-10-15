"""Data and protocol regression tests for V2; use source-only synthetic inputs."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from v2_experiment import (raw_cycles, calibrated_noise, fresh_noisy_source,
                           fixed_noisy_validation, summarize)


def fixture_source():
    t=np.linspace(0,1,10001)
    y=np.stack([np.sin(2*np.pi*(i+1)*19*t) for i in range(5)],axis=-1)
    return {"a":(t,y,np.full_like(t,2*np.pi*19))}


def test_raw_cycle_interpolation_and_shape():
    raw=fixture_source()
    meta=[{"record_id":"a","label":2,"start":.12,"end":.36}]
    x=raw_cycles(raw,meta)
    assert x.shape==(1,5,4096) and x.dtype==np.float32
    assert np.isfinite(x).all()
    assert x.max() <= 1.01 and x.min() >= -1.01


def test_calibrated_noise_is_per_record_per_phase():
    raw=fixture_source()
    noisy=calibrated_noise(raw,{"a":10},lambda rid:np.random.default_rng(5))["a"]
    old=raw["a"][1];new=noisy[1]
    measured=10*np.log10(np.mean(old**2,axis=0)/np.mean((new-old)**2,axis=0))
    assert np.allclose(measured,10,atol=1e-9)
    assert np.array_equal(raw["a"][0],noisy[0])
    assert np.array_equal(raw["a"][2],noisy[2])


def test_epoch_noise_is_deterministic():
    raw=fixture_source()
    meta=[{"record_id":"a","label":2,"start":.12,"end":.36}]
    first=fresh_noisy_source(raw,meta,42,1)
    again=fresh_noisy_source(raw,meta,42,1)
    new=fresh_noisy_source(raw,meta,42,2)
    assert np.array_equal(first,again)
    assert not np.array_equal(first,new)


def test_fixed_val_noise_is_repeatable_and_source_only():
    raw=fixture_source()
    meta=[{"record_id":"a","label":2,"start":.12,"end":.36}]
    va1=fixed_noisy_validation(raw,meta)
    va2=fixed_noisy_validation(raw,meta)
    assert set(va1)=={10,20,30}
    assert all(np.array_equal(va1[k],va2[k]) for k in va1)


def test_summarizer_never_accepts_final_holdout(tmp_path):
    path=tmp_path/"evaluations"/"identity"/"seed42"
    path.mkdir(parents=True)
    item={"method":"identity","condition":"clean","final_holdout_evaluated":True,
          "metrics":{"macro_f1":.5,"accuracy":.5,"normal_false_alarm_rate":.1,"missed_fault_as_normal":1}}
    (path/"target_dev_clean.json").write_text(json.dumps(item))
    with pytest.raises(ValueError, match="Invalid result"):
        summarize(tmp_path,["identity"],[42],[None])
