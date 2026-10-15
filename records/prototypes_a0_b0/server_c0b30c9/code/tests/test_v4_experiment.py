"""V4 result protocol and source-only integration guards."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import v4_experiment as v4
from v4_hpasf import METHODS


def test_protocol_and_controls_are_frozen():
    assert METHODS == ("lowpass3","lowpass4","static4","hpasf")
    assert v4.DEFAULT_SEEDS == [42,43,44]
    assert v4.DEFAULT_CONDITIONS == [None,10,15,20,25,30,35,40]
    assert "source" in v4.SOURCE_SELECTION_RULE
    assert "target" not in v4.SOURCE_SELECTION_RULE


def test_synthetic_smoke_runs_end_to_end():
    v4.smoke()


def test_summary_recomputes_paired_differences(tmp_path):
    for mode in METHODS:
        for seed in [42,43]:
            for cond in ["clean","snr10"]:
                path=tmp_path/"evaluations"/mode/f"seed{seed}"/f"target_dev_{cond}.json"
                path.parent.mkdir(parents=True,exist_ok=True)
                f1={"lowpass3":.95,"lowpass4":.94,"static4":.96,"hpasf":.97}[mode]
                path.write_text(json.dumps({"method":mode,"seed":seed,"condition":cond,
                    "record_families":88,"windows":2474,"final_holdout_evaluated":False,
                    "metrics":{"macro_f1":f1,"accuracy":f1,"normal_false_alarm_rate":0,
                                "missed_fault_as_normal":0}}))
    out=v4.summarize(tmp_path,list(METHODS),[42,43],[None,10])
    assert len(out["aggregates"]) == 8
    pair=next(r for r in out["paired_vs_lowpass3"]
              if r["method_minus_lowpass3"]=="hpasf" and r["condition"]=="snr10")
    assert pair["mean_delta_pp"] == pytest.approx(2)
    assert pair["positive_seeds"] == 2
    assert (tmp_path/"PER_SEED_RESULTS.csv").exists()


def test_bad_result_provenance_fails(tmp_path):
    path=tmp_path/"evaluations"/"lowpass3"/"seed42"/"target_dev_clean.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"method":"lowpass3","seed":42,"condition":"clean",
                                "record_families":88,"windows":2474,"final_holdout_evaluated":True,
                                "metrics":{"macro_f1":1.0}}))
    with pytest.raises(ValueError,match="provenance"):
        v4.summarize(tmp_path,["lowpass3"],[42],[None])


def test_parameter_validation_prevents_invalid_runs(tmp_path):
    with pytest.raises(ValueError):
        v4.train("hpasf",42,tmp_path,epochs=0)
    with pytest.raises(ValueError):
        v4.train("fake",42,tmp_path)
    destination=tmp_path/"models"/"hpasf"/"seed42"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        v4.train("hpasf",42,tmp_path)
