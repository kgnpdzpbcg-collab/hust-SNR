"""检查固定数据池、漏报方向及配对统计，测试不导入Torch或执行训练。"""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_v4_strong_snr import METHODS, SEEDS, LEVELS, frozen_ids, summarize


def results():
    """构造三种子差异不相同的结果，防止错误地对标准差直接作差。"""
    rows = []
    for method_index, method in enumerate(METHODS):
        for index, seed in enumerate(SEEDS):
            for level in LEVELS:
                value = .90 + .01*method_index - (10-level)*(.002 + .001*index)
                cm = [[0]*6 for _ in range(6)]
                for label in range(6):
                    cm[label][label] = 1
                cm[1][0] = index
                # cm[0][1]故意不同，检查故障漏报和健康误报不会互换。
                cm[0][1] = 9
                metrics = {"macro_f1": value, "accuracy": value, "balanced_accuracy": value,
                           "normal_false_alarm_rate": .01, "missed_fault_as_normal": index,
                           "normal_false_alarms": 9, "per_class_f1": [value]*6,
                           "per_class_recall": [value]*6, "confusion_matrix": cm}
                rows.append({"method": method, "seed": seed, "snr_db": level,
                             "truth": list(range(6)), "metrics": metrics})
    return rows


def test_frozen_record_guard_rejects_holdout_and_changed_window_count():
    """即使总数量相同，替换成保留编号也应拒绝。"""
    ids = [f"record{i}" for i in range(88)]
    split = {"target_dev_records": ids, "target_holdout_records": ["holdout"]}
    metadata = [{"record_id": ids[i % 88]} for i in range(2474)]
    assert frozen_ids(split, metadata) == sorted(ids)
    with pytest.raises(ValueError):
        frozen_ids(split, metadata[:-1])
    wrong = copy.deepcopy(metadata)
    wrong[0]["record_id"] = "holdout"
    with pytest.raises(ValueError):
        frozen_ids(split, wrong)


def test_drop_is_paired_and_false_healthy_uses_fault_row():
    """5dB退化按种子分别为1/1.5/2pp，漏报读取cm[故障][健康]。"""
    summary = summarize(results())
    drop = next(r for r in summary["drop_from_10db"] if r["method"] == "lowpass3" and r["snr_db"] == 5)
    assert drop["macro_f1_drop_pp"]["values"] == pytest.approx([1, 1.5, 2])
    assert drop["macro_f1_drop_pp"]["mean"] == pytest.approx(1.5)
    assert drop["macro_f1_drop_pp"]["std"] == pytest.approx(.5)
    row = next(r for r in summary["per_class"] if r["method"] == "lowpass3" and r["snr_db"] == 5 and r["label"] == 1)
    assert row["fault_as_healthy"]["values"] == [0, 1, 2]
    pair = next(r for r in summary["paired_methods"] if r["left"] == "hpasf" and r["right"] == "lowpass4" and r["snr_db"] == 0)
    assert pair["macro_f1_delta_pp"]["values"] == pytest.approx([2, 2, 2])


def test_missing_seed_and_different_truth_are_rejected():
    """不得漏掉低分种子，也不得混用不同周期标签。"""
    with pytest.raises(ValueError, match="完整36"):
        summarize(results()[:-1])
    wrong = results()
    wrong[-1]["truth"] = [0, 1, 2, 3, 5, 4]
    with pytest.raises(ValueError, match="同一批周期"):
        summarize(wrong)
