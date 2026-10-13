"""独立复算488组诊断指标、记录级错误和原权重来源；只导出客观数字。"""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

ROOT = Path(__file__).resolve().parent


def read(path):
    """读取原始精度JSON，核验不得修改原预测。"""
    return json.loads(path.read_text())


def metrics(truth, prediction):
    """sklearn独立实现六类指标，混淆矩阵行真实列预测。"""
    cm = confusion_matrix(truth, prediction, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=range(6), zero_division=0)
    fp = int(cm[0, 1:].sum())
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "confusion_matrix": cm.tolist(), "normal_false_alarms": fp,
            "normal_false_alarm_rate": fp/int(cm[0].sum()), "missed_fault_as_normal": int(cm[1:, 0].sum())}


def sha(path):
    """SHA核验服务器原字节和旧输入/结果保护清单。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """所有检查通过才生成导出包，不包括PT快照或原始电流。"""
    assert read(ROOT/"results/STATUS.json")["status"] == "passed" and read(ROOT/"trajectory/STATUS.json")["status"] == "passed"
    diagnostic = read(ROOT/"results/TARGET_DIAGNOSTICS.json")
    scan = read(ROOT/"trajectory/EPOCH_TARGET_DETAILS.json")
    assert len(diagnostic) == 88 and len(scan["results"]) == 400
    for row in diagnostic + [{**r, "truth": scan["truth"]} for r in scan["results"]]:
        actual = metrics(row["truth"], row["predicted"])
        for key, value in actual.items():
            np.testing.assert_allclose(value, row["metrics"][key], atol=1e-12, rtol=0)
    extra = ROOT.parents[2]/"experiments/five_models_extra_seeds/3f07228"
    meta = read(extra/"code/outputs/gate0_v1/target_dev_clean_windows.json")
    lookup = {(r["seed"],r["variant"],r["condition"]): r for r in diagnostic}
    truth = np.array(diagnostic[0]["truth"])
    with (ROOT/"results/PER_RECORD_ERRORS.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 88*88
    for row in rows:
        result = lookup[int(row["seed"]),row["variant"],row["condition"]]
        pred = np.array(result["predicted"])
        mask = np.array([r["record_id"] == row["record_id"] for r in meta])
        fault = mask & (truth>0)
        assert int(row["errors"]) == int((pred[mask] != truth[mask]).sum())
        assert int(row["fault_errors"]) == int((pred[fault] != truth[fault]).sum())
        assert int(row["cycles"]) == int(mask.sum()) and int(row["fault_cycles"]) == int(fault.sum())
        assert [int(row[f"fault_predicted_{k}"]) for k in range(6)] == np.bincount(pred[fault],minlength=6).tolist()
    assert all(sha(Path(path)) == expected for path,expected in read(ROOT/"results/PROTECTED_FILES_SHA256.json").items())
    weights = read(ROOT/"results/STATUS.json")["weights_sha256"]
    assert all(sha(Path(path)) == expected for path, expected in weights.items())
    status = {"status": "passed", "independently_recomputed_target_metric_sets": 488,
              "record_error_rows_verified": len(rows), "native_prediction_replays": 24,
              "trajectory_exact_histories": 100, "original_best_weight_states_exact": 2,
              "official_results_replaced": False, "final_holdout_evaluated": False,
              "code_sha256": {name: sha(ROOT/name) for name in ("v5_low_seed_diagnosis.py", "v5_replay_diagnosis.py")}}
    (ROOT/"VERIFICATION.json").write_text(json.dumps(status, ensure_ascii=False, indent=2)+"\n")
    files = [p for folder in (ROOT/"results",ROOT/"trajectory") for p in folder.rglob("*") if p.is_file() and p.suffix in (".json",".csv")]
    files += [ROOT/name for name in ("v5_low_seed_diagnosis.py", "v5_replay_diagnosis.py", "VERIFY_AND_EXPORT.py", "VERIFICATION.json")]
    manifest = {"status": "passed", "files": {p.relative_to(ROOT).as_posix(): {"sha256":sha(p),"bytes":p.stat().st_size} for p in files},
                "raw_current_uploaded": False, "weights_or_snapshots_uploaded": False, "environment_uploaded": False}
    (ROOT/"RESULTS_EXPORT_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with tarfile.open(ROOT/"diagnosis_records.tar.gz", "w:gz") as archive:
        for p in files+[ROOT/"RESULTS_EXPORT_MANIFEST.json"]:
            archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print("DIAGNOSIS_INDEPENDENT_VERIFICATION_AND_EXPORT_PASSED",len(files),"files",flush=True)


if __name__ == "__main__":
    main()
