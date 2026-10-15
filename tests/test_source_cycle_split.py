"""验证新周期划分的标签来源、时间隔离、重复检测及训练入口版本选择。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import source_cycle_split as split


def synthetic_records():
    """构造共享健康片段、五类不同故障的五个记录，避免依赖用户私有MAT。"""
    time = np.arange(10000) / 10000.0
    common = np.stack([np.sin(2 * np.pi * 50 * time + i * 0.3) + 0.1 * time
                       for i in range(5)], axis=1)
    records, labels = {}, {}
    for fault in range(1, 6):
        rid = f"S-600-0-{fault}-O"
        currents = common.copy()
        currents[time >= 0.5] *= 1 + fault * 0.1
        records[rid] = (time, currents, np.full_like(time, 2 * np.pi * 50))
        labels[rid] = split.filename_label(rid)
    return records, labels


def test_all_four_missing_labels_follow_verified_filename_convention():
    """四个补录文件仍明确属于文件名推定，编号沿用作者既有映射。"""
    assert [split.filename_label(f"S-400-0-{i}-O")["fault_label"] for i in (1, 2, 3)] == [1, 2, 3]
    assert split.filename_label("S-1000-9-4-O")["fault_name"] == "OUS(A)&OUS(B)"
    with pytest.raises(ValueError):
        split.filename_label("S-400-0-6-O")


@pytest.mark.parametrize("count", [3, 7, 17, 22])
def test_time_partition_preserves_one_whole_cycle_gap(count):
    train_end, val_start = split.split_positions(count, 0.2, 1)
    assert 0 < train_end < val_start < count
    assert val_start - train_end == 1
    with pytest.raises(ValueError):
        split.split_positions(2, 0.2, 1)


def test_all_conditions_are_shared_but_health_copies_are_not_repeated():
    records, labels = synthetic_records()
    config = split.read(split.CONFIG)
    parts, families = split.make_windows(records, labels, config)
    assert len(families) == 1
    for part in ("train", "val"):
        assert {r["record_id"] for r in parts[part]} == set(records)
        assert {r["record_id"] for r in parts[part] if r["state"] == 0} == {min(records)}
    verified = split.verify_split(records, labels, parts, config)
    assert verified["status"] == "passed"
    assert verified["cross_split_exact_cycle_duplicates"] == 0
    assert verified["segments_checked"] == 6


def test_validation_window_cannot_overlap_training():
    records, labels = synthetic_records()
    config = split.read(split.CONFIG)
    parts, _ = split.make_windows(records, labels, config)
    first = parts["train"][0]
    validation = next(r for r in parts["val"] if (r["record_id"], r["state"]) == (first["record_id"], first["state"]))
    validation["start"] = first["start"]
    with pytest.raises(ValueError, match="时间范围重叠"):
        split.verify_split(records, labels, parts, config)


def test_nonoverlapping_but_identical_cycles_are_rejected():
    records, labels = synthetic_records()
    # 时间不相交并不足以保证无重复；常数波形可以在任意窗口完全重复。
    records = {rid: (t, np.ones_like(x), w) for rid, (t, x, w) in records.items()}
    config = split.read(split.CONFIG)
    parts, _ = split.make_windows(records, labels, config)
    with pytest.raises(ValueError, match="跨集合完全重复周期"):
        split.verify_split(records, labels, parts, config)


def test_profile_metadata_cannot_be_silently_changed(tmp_path):
    metadata = tmp_path / "source_train_windows.json"
    split.save(metadata, [])
    split.save(tmp_path / "SOURCE_SPLIT.json", {
        "protocol": split.PROFILE, "source_records": 63, "sets": {},
        "metadata_sha256": {metadata.name: split.sha(metadata)},
    })
    assert split.source_provenance(tmp_path)["protocol"] == split.PROFILE
    split.save(metadata, [{"label": 5}])
    with pytest.raises(ValueError, match="指纹改变"):
        split.source_provenance(tmp_path)


def test_cli_defaults_to_new_split_without_starting_training(monkeypatch):
    """替换训练函数为记录器，只检查参数路由，不做参数更新。"""
    import hust_a1_b1_experiment as experiment
    calls = []
    monkeypatch.setattr(experiment, "train", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["experiment", "train", "--methods", "a1", "--seeds", "42"])
    experiment.main()
    assert calls[0][2] == experiment.CYCLE_OUT
    assert calls[0][4] == split.DEFAULT_OUT
    assert calls[0][3] is None


def test_legacy_split_requires_explicit_cli_choice(monkeypatch, tmp_path):
    import hust_a1_b1_experiment as experiment
    calls = []
    monkeypatch.setattr(experiment, "train", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["experiment", "train", "--methods", "a1", "--seeds", "42",
                                      "--legacy-source-split", "--source-strong-root", str(tmp_path)])
    experiment.main()
    assert calls[0][2] == experiment.OUT
    assert calls[0][4] is None


def test_new_cli_does_not_silently_ignore_legacy_data_argument(monkeypatch, tmp_path):
    import hust_a1_b1_experiment as experiment
    monkeypatch.setattr(sys, "argv", ["experiment", "train", "--source-strong-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        experiment.main()


def test_summary_rejects_mixed_source_splits(tmp_path):
    import hust_a1_b1_experiment as experiment
    for method, protocol in (("a1", split.PROFILE), ("b1", "gate0_v1")):
        folder = tmp_path / f"evaluations/{method}/seed42"
        folder.mkdir(parents=True)
        split.save(folder / "target_dev_snr5.json", {
            "method": method, "seed": 42, "protocol": experiment.PROTOCOL,
            "condition": "snr5", "record_families": 88, "windows": 2474,
            "final_holdout_evaluated": False, "truth": [0],
            "source_split": {"protocol": protocol},
            "metrics": {"macro_f1": 1, "accuracy": 1, "missed_fault_as_normal": 0,
                        "normal_false_alarms": 0, "per_class_f1": [1] * 6, "per_class_recall": [1] * 6},
        })
    with pytest.raises(ValueError, match="不同仿真划分"):
        experiment.summarize(tmp_path, ["a1", "b1"], [42], [5])
