"""在基础审计之后复核完整实测波形重复、工况冲突及重复时间戳。

仅使用hust缓存和审计标签，不修改数据，也不自动执行去重/数据划分。
输出的435等数量均由数据计算，不能用文件名硬编码故障或工况。
"""
from pathlib import Path
from collections import Counter, defaultdict
import json
import numpy as np
from audit_dataset import read_matrix, array_sha


def main():
    """健康片段和完整记录分别核对，不能把相同时间戳等同于相同电流。"""
    work = Path(__file__).resolve().parents[1]
    base = work/"analysis/data_audit_v1"
    audit = json.loads((base/"DATA_AUDIT.json").read_text(encoding="utf-8"))
    labels = json.loads((base/"LABELS.json").read_text(encoding="utf-8"))
    cross_conditions = {}
    for kind in ("simulation", "experimental"):
        cross_conditions[kind] = []
        for group in audit[kind]["exact_prefault_duplicate_groups"]:
            conditions = sorted({(labels[kind][i]["speed"], labels[kind][i]["torque"]) for i in group})
            if len(conditions)>1:
                cross_conditions[kind].append({"record_ids":group,"conditions":conditions})
    complete = defaultdict(list)
    all_column_hash = {}
    difference_min, difference_max = float("inf"), 0.
    nonidentical, repeated = 0, 0
    for name in sorted(labels["experimental"]):
        value = read_matrix(Path(audit["experimental"]["cache"])/(name+".mat"), "name")
        complete[array_sha(value[:,1:6])].append(name)
        all_column_hash[name] = array_sha(value)
        # 同一时间戳的两行大多有不同电流，不能直接把第二行当重复波形删除。
        indices = np.flatnonzero(np.diff(value[:,0]) == 0)
        for index in indices:
            difference = float(np.max(np.abs(value[index+1,1:6]-value[index,1:6])))
            difference_min = min(difference_min, difference)
            difference_max = max(difference_max, difference)
            nonidentical += int(difference>0)
            repeated += 1
    groups = [names for names in complete.values() if len(names)>1]
    conflicts = []
    canonical = []
    for names in complete.values():
        fault_labels = {labels["experimental"][name]["fault_label"] for name in names}
        assert len(fault_labels)==1, "同一电流对应冲突故障标签，不能继续构造六分类"
        canonical.append(min(names))
        if len(names)>1:
            values = [read_matrix(Path(audit["experimental"]["cache"])/(name+".mat"), "name") for name in names]
            conflicts.append({"record_ids":names,
                              "all_seven_columns_equal":len({all_column_hash[name] for name in names})==1,
                              "time_column_equal":all(np.array_equal(values[0][:,0],value[:,0]) for value in values),
                              "last_column_max_absolute_difference":max(float(np.max(np.abs(values[0][:,-1]-value[:,-1]))) for value in values),
                              "torques":sorted({labels["experimental"][name]["torque"] for name in names}),
                              "fault_label":next(iter(fault_labels))})
    result = {"status":"verified", "exact_complete_current_duplicate_groups":groups,
              "duplicate_metadata_comparison":conflicts, "prefault_duplicate_groups_cross_conditions":cross_conditions,
              "unique_complete_current_records":len(complete),
              "canonical_fault_record_counts":dict(Counter(labels["experimental"][name]["fault_label"] for name in canonical)),
              "canonical_record_ids_if_first_lexicographic_representative_adopted":sorted(canonical),
              "dedup_applied_to_raw_files":False,"split_created":False,
              "repeated_time_row_pairs":repeated,"nonidentical_repeated_time_row_pairs":nonidentical,
              "repeated_time_current_difference_min":difference_min,"max":difference_max,
              "timestamp_duplicates_are_not_duplicate_current_rows":True}
    (base/"DUPLICATE_FOLLOWUP.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","unique_complete_current_records","canonical_fault_record_counts",
                                         "repeated_time_row_pairs","nonidentical_repeated_time_row_pairs")},ensure_ascii=False))


if __name__ == "__main__":
    main()
