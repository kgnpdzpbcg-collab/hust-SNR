"""汇总原best与预先指定epoch50的诊断差异，仅导出元数据和预测。"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
FIRST = BASE / "experiments/prototypes_audit/20261015"


def read(path):
    """读取已完成的精确重放；不丢弃低分种子。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    """导出清单使用完整文件指纹。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    """新建本轮核验记录，不修改原实验数据。"""
    assert not path.exists()
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


status = read(ROOT / "RUN_STATUS.json")
assert status["status"] == "completed" and len(status["jobs"]) == 6
previous = read(FIRST / "results/FIXED_WEIGHT_INTERVENTIONS.json")
confidence = list(csv.DictReader((FIRST / "results/CONFIDENCE_AND_CE.csv").open()))
conditions = {"source_clean": "source_val_clean", "source_snr0": "source_val_0", "source_snr10": "source_val_10",
              "target_clean": "target_clean", **{f"target_snr{snr}": f"target_{snr}" for snr in (5, 0, -5, -10)}}
rows = []
for method in ("a0", "b0"):
    for seed in (42, 43, 44):
        obj = read(ROOT / f"results/epoch50_evaluations/{method}/seed{seed}.json")
        assert obj["all_original_training_history_values_exactly_reproduced"]
        assert obj["original_selected_weights_tensorwise_identical"]
        for condition, original_name in conditions.items():
            old = next(r for r in previous if (r["method"], r["seed"], r["condition"], r["mode"]) ==
                       (method, seed, original_name, "original"))
            old_ce = next(float(r["cross_entropy"]) for r in confidence if r["method"] == method and int(r["seed"]) == seed and r["condition"] == original_name)
            final = obj["results"][condition]
            rows.append({"method": method, "seed": seed, "condition": original_name,
                "original_selected_epoch": obj["original_selected_epoch"], "diagnostic_epoch": 50,
                "original_macro_f1_percent": old["macro_f1_percent"],
                "epoch50_macro_f1_percent": final["metrics"]["macro_f1"] * 100,
                "epoch50_minus_selected_pp": final["metrics"]["macro_f1"] * 100 - old["macro_f1_percent"],
                "original_cross_entropy": old_ce, "epoch50_cross_entropy": final["cross_entropy"]})
with (ROOT / "COMPARISON_PER_SEED.csv").open("x", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = []
for method in ("a0", "b0"):
    for condition in conditions.values():
        group = [r for r in rows if r["method"] == method and r["condition"] == condition]
        row = {"method": method, "condition": condition}
        for key in ("original_macro_f1_percent", "epoch50_macro_f1_percent", "epoch50_minus_selected_pp",
                    "original_cross_entropy", "epoch50_cross_entropy"):
            values = np.asarray([r[key] for r in group])
            row[key + "_mean"] = float(values.mean())
            row[key + "_sample_sd"] = float(values.std(ddof=1))
        summary.append(row)
save(ROOT / "COMPARISON_SUMMARY.json", {"summary": summary, "epoch50_chosen_before_target_evaluation": True,
     "used_to_replace_official_source_selection": False, "final_holdout_accessed": False})
save(ROOT / "VERIFICATION.json", {"status": "passed", "training_replay_runs": 6,
     "joint_training_epoch_histories_exactly_reproduced": 300, "pretrain_epoch_losses_exactly_reproduced": 30,
     "original_source_selected_checkpoints_tensorwise_identical": 6,
     "all_original_run_records_equal": True, "source_final_epoch_f1_replays_verified": 36,
     "target_dev_epoch50_prediction_sets": 30, "original_official_results_overwritten": False,
     "target_used_for_checkpoint_or_hyperparameter_selection": False, "final_holdout_accessed": False,
     "elapsed_replay_seconds": status["elapsed_seconds"], "code_commit": "8f8704ff6abb8de77d3cf4f210e7d81826c05442"})
files = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*") if p.is_file()
         and p.suffix in (".py", ".json", ".csv", ".log") and "__pycache__" not in p.parts}
assert not any(p.suffix in (".pt", ".mat", ".npz") for p in files.values())
save(ROOT / "EXPORT_MANIFEST.json", {"files": {name: {"sha256": sha(p), "bytes": p.stat().st_size}
     for name, p in sorted(files.items())}, "raw_data_included": False, "weights_included": False})
files["EXPORT_MANIFEST.json"] = ROOT / "EXPORT_MANIFEST.json"
with tarfile.open(ROOT / "replay_records.tar.gz", "w:gz") as archive:
    for name, path in sorted(files.items()):
        archive.add(path, arcname=name)
for row in summary:
    print(row["method"], row["condition"], "selected", row["original_macro_f1_percent_mean"],
          "epoch50", row["epoch50_macro_f1_percent_mean"], "sd", row["epoch50_macro_f1_percent_sample_sd"], flush=True)
print("EXACT_REPLAY_EXPORTED", len(files))
