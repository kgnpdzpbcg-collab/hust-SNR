"""独立重载MAT和权重，重放36次推理，并用sklearn复算全部指标。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
record = json.loads((OUT / "RUN_RECORD.json").read_text())
V4 = Path(record["v4_run_root"])
STRONG = Path(record["strong_data_root"])
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(V4 / "code/scripts"))
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import v2_experiment as v2
from v1_noise_consistency import fix_randomness
from v4_hpasf import HPASFCNN
from evaluate_v4_strong_snr import sha, summarize, METHODS, SEEDS, LEVELS


def read(path):
    """本核查使用实际服务器文件，不使用聊天中展示的四舍五入数值。"""
    return json.loads(path.read_text())


def assert_close(left, right):
    """逐字段比较，允许不同统计库产生不超过1e-10的浮点舍入差。"""
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_close(a, b)
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=0)
    else:
        assert left == right


started = time.perf_counter()
metadata = read(V4 / "code/outputs/gate0_v1/target_dev_clean_windows.json")
ids = sorted({r["record_id"] for r in metadata})
split = read(V4 / "code/configs/splits_gate0_v1.json")
assert ids == sorted(split["target_dev_records"]) and not set(ids) & set(split["target_holdout_records"])
assert len(metadata) == 2474 and len(ids) == 88
truth = np.array([r["label"] for r in metadata])
inputs = {}
max_snr_error, generated_mats = 0., 0
for level in LEVELS:
    raw = {}
    for rid in ids:
        # 实测MAT字段含时间、五相电流、角速度；只检查/加噪五相电流。
        clean = loadmat(V4 / "code/data/target_dev/clean" / f"{rid}.mat")["name"]
        author = loadmat(V4 / "target_snr/SNR_10" / f"{rid}.mat")["I_noisy"]
        array = (author if level == 10 else loadmat(STRONG / f"generated/target/SNR_{level}/{rid}.mat")["I_noisy"])
        assert array.shape == (10031, 7) and array.dtype == np.float64 and np.isfinite(array).all()
        assert np.array_equal(array[:, [0, -1]], clean[:, [0, -1]])
        if level != 10:
            expected = clean[:, 1:6] + (author[:, 1:6] - clean[:, 1:6])*10**((10-level)/20)
            assert np.array_equal(array[:, 1:6], expected)
            actual = 10*np.log10(np.mean(clean[:, 1:6]**2, axis=0) /
                                 np.mean((array[:, 1:6]-clean[:, 1:6])**2, axis=0))
            max_snr_error = max(max_snr_error, float(np.max(np.abs(actual-level))))
            assert np.max(np.abs(actual-level)) < 1e-10
            generated_mats += 1
        t, index, count = np.unique(array[:, 0], return_index=True, return_counts=True)
        merged = np.add.reduceat(array[:, 1:], index, axis=0) / count[:, None]
        assert len(t) == 10011
        raw[rid] = (t, merged[:, :5], merged[:, -1])
    inputs[level] = v2.raw_cycles(raw, metadata)

recomputed = []
for method in METHODS:
    for seed in SEEDS:
        fix_randomness(seed)
        checkpoint = torch.load(V4 / f"results/models/{method}/seed{seed}/best.pt", map_location="cpu", weights_only=True)
        model = HPASFCNN(method).cuda()
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        for level in LEVELS:
            saved = read(OUT / f"evaluations/{method}/seed{seed}/target_dev_snr{level}.json")
            assert saved["truth"] == truth.tolist()
            # 重新加载权重、直接分批前向，独立于运行入口的inference包装函数。
            predictions = []
            with torch.inference_mode():
                for index in range(0, len(truth), 128):
                    scores = model(torch.from_numpy(inputs[level][index:index+128]).cuda())
                    predictions.extend(scores.argmax(1).cpu().tolist())
            assert predictions == saved["predicted"]
            matrix = confusion_matrix(truth, predictions, labels=range(6))
            _, recall, f1, _ = precision_recall_fscore_support(truth, predictions, labels=range(6), zero_division=0)
            false_alarm = int(matrix[0, 1:].sum())
            metrics = {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, predictions)),
                       "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(),
                       "per_class_recall": recall.tolist(), "confusion_matrix": matrix.tolist(),
                       "normal_false_alarms": false_alarm,
                       "normal_false_alarm_rate": false_alarm / int(matrix[0].sum()),
                       "missed_fault_as_normal": int(matrix[1:, 0].sum())}
            assert_close(saved["metrics"], metrics)
            if level == 10:
                previous = read(V4 / f"results/evaluations/{method}/seed{seed}/target_dev_snr10.json")
                assert previous["predicted"] == predictions and previous["truth"] == saved["truth"]
            recomputed.append({**saved, "metrics": metrics})
        assert all(torch.equal(value.cpu(), checkpoint["state"][name]) for name, value in model.state_dict().items())
assert_close(read(OUT / "SUMMARY.json"), summarize(recomputed))
for relative, count in (("PER_SEED_RESULTS.csv", 36), ("PER_CLASS_PER_SEED.csv", 216), ("PAIRED_METHODS.csv", 54)):
    with (OUT / relative).open() as stream:
        assert len(list(csv.DictReader(stream))) == count
for path, expected in record["protected_files_unchanged"].items():
    assert sha(Path(path)) == expected
assert sha(BASE / "scripts/evaluate_v4_strong_snr.py") == record["evaluation_script_sha256"]
verified = {"status": "passed", "prediction_vectors_exactly_reproduced": 36,
            "metric_sets_independently_recomputed_with_sklearn": 36,
            "original_10db_predictions_exactly_reproduced": 12,
            "new_target_5_0_mat_files_verified": generated_mats,
            "max_full_record_per_phase_snr_error_db": max_snr_error,
            "model_state_dicts_unchanged": 12,
            "protected_files_unchanged": len(record["protected_files_unchanged"]),
            "aggregate_groups_verified": 12, "per_class_groups_verified": 72,
            "paired_method_groups_verified": 18, "degradation_groups_verified": 8,
            "final_holdout_evaluated": False, "training_updates": 0,
            "verification_seconds": time.perf_counter() - started,
            "verifier_sha256": sha(Path(__file__))}
(OUT / "VERIFICATION.json").write_text(json.dumps(verified, indent=2) + "\n")
print(json.dumps(verified, indent=2), flush=True)
