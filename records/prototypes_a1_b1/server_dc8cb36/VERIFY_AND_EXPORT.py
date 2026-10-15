"""独立复算A1/B1源选模、交叉熵和全部实测预测；导出客观元数据，排除原波形和权重。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import tarfile
import time

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE, OUT = ROOT / "code", ROOT / "results"
sys.path.insert(0, str(CODE / "scripts"))
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import hust_a1_b1_experiment as exp
from prepare import OUT as PREPARED


def read(path):
    """读取未舍入的原记录，不能跳过或替换低分种子。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """验证原文件及发布字节；模型权重只在服务器使用，不打包。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    """指标只允许浮点末位误差；预测必须逐项相同。"""
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def metric(truth, predicted):
    """独立sklearn复算六类指标，混淆矩阵行真实、列预测。"""
    matrix = confusion_matrix(truth, predicted, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, predicted, labels=range(6), zero_division=0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(),
            "per_class_recall": recall.tolist(), "normal_false_alarms": int(matrix[0, 1:].sum()),
            "normal_false_alarm_rate": float(matrix[0, 1:].sum() / matrix[0].sum()),
            "missed_fault_as_normal": int(matrix[1:, 0].sum()), "confusion_matrix": matrix.tolist()}


def main():
    """仅在6组正式训练和24组实测评分完成后校验，全部通过才导出。"""
    started = time.perf_counter()
    status, deploy, preflight = [read(ROOT / name) for name in ("RUN_STATUS.json", "DEPLOYMENT_MANIFEST.json", "PREFLIGHT.json")]
    assert status["status"] == "completed" and len(status["jobs"]) == 8
    protected = read(ROOT / "PROTECTED_FILES_SHA256.json")
    assert all(sha(path) == expected for path, expected in protected.items())
    assert all(sha(CODE / name) == expected for name, expected in deploy["code_sha256"].items())
    models, selections, weights, history_rows, warmup_rows = {}, [], {}, [], []
    (_, tm, _, _), (_, vm, vy, val_clean, val_noise) = exp.source_views(Path(deploy["source_strong_root"]))
    values = {"clean": val_clean, **{f"snr{snr}": x for snr, x in val_noise.items()}}
    source_replay = []
    for method in exp.METHODS:
        for seed in exp.SEEDS:
            folder = OUT / f"models/{method}/seed{seed}"
            run, history = read(folder / "run.json"), read(folder / "history.json")
            assert (run["method"], run["seed"], run["protocol"]) == (method, seed, exp.PROTOCOL)
            assert run["config"] == exp.get_config()
            assert not run["target_used_for_training_or_selection"] and not run["final_holdout_evaluated"]
            assert (run["source_train_cycles"], run["source_val_cycles"]) == (738, 302)
            assert run["parameters"] == preflight["parameter_counts"][method]
            assert run["initial_parameters_sha256"] == preflight["initial_parameters_sha256"][f"{method}/seed{seed}"]
            assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
            warmup = run["pretrain_recovery_losses"]
            assert len(warmup) == (10 if method == "b1" else 0)
            for epoch, loss in enumerate(warmup, 1):
                assert np.isfinite(loss) and loss >= 0
                warmup_rows.append({"method": method, "seed": seed, "pretrain_epoch": epoch, "source_recovery_loss": loss})
            for row in history:
                assert np.isfinite(row["train_loss_mean"]) and row["train_loss_mean"] >= 0
                assert set(row["noisy_f1"]) == {"0", "5", "10", "20", "30"}
                close(row["noise_mean_f1"], np.mean(list(row["noisy_f1"].values())))
                if method == "b1":
                    assert row["source_restore_loss_mean"] >= 0
                    assert abs(row["train_loss_mean"] - row["classification_ce_mean"] - row["source_restore_loss_mean"]) < 1e-6
                history_rows.append({"method": method, "seed": seed, **row})
            # 独立按干净验证1个百分点候选带及五档均值排序，增加CE平局判据，不调用原选模函数。
            peak = max(row["clean_f1"] for row in history)
            eligible = [r for r in history if r["clean_f1"] >= peak - .01 - 1e-12]
            best = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["noise_mean_ce"], r["epoch"]))[0]
            assert best == run["selected"]
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            assert checkpoint["selected"] == best and checkpoint["config"] == exp.get_config()
            assert checkpoint["source_only"] and not checkpoint["final_holdout_evaluated"]
            if method == "b1":
                assert all(int(value) == best["epoch"] * 24 for name, value in checkpoint["state"].items() if name.endswith("num_batches_tracked"))
            exp.fix_randomness(seed)
            model = exp.build_model(method).cuda().eval().requires_grad_(False)
            model.load_state_dict(checkpoint["state"], strict=True)
            for condition, x in values.items():
                predicted = exp.infer(model, x, torch.device("cuda"))
                score = metric(vy, predicted)
                expected = best["clean_f1"] if condition == "clean" else best["noisy_f1"][condition[3:]]
                close(score["macro_f1"], expected)
                ce = exp.source_cross_entropy(model, x, vy, torch.device("cuda"))
                if condition != "clean":
                    np.testing.assert_allclose(ce, best["noisy_ce"][condition[3:]], atol=1e-7, rtol=0)
                source_replay.append({"method": method, "seed": seed, "condition": condition,
                                      "macro_f1_percent": 100 * score["macro_f1"], "cross_entropy": ce})
            selections.append({"method": method, "seed": seed, "selected_epoch": best["epoch"],
                               "eligible_epoch_count": len(eligible), "source_clean_f1_percent": 100 * best["clean_f1"],
                               "source_five_noise_mean_f1_percent": 100 * best["noise_mean_f1"]})
            models[method, seed] = (model, checkpoint["state"])
            weights[str(folder / "best.pt")] = sha(folder / "best.pt")
    # 逐周期重放24组实际预测，并与原A0/B0及LP4＋ResNet同种子结果作客观配对。
    frozen = read(PREPARED / "target_dev_clean_windows.json")
    truth = np.asarray([r["label"] for r in frozen], dtype=np.int64)
    assert np.bincount(truth, minlength=6).tolist() == [1106, 307, 297, 298, 237, 229]
    reference_roots = {5: BASE / "experiments/architecture_suite/5c75bea/results",
                       0: BASE / "experiments/architecture_suite/5c75bea/results",
                       -5: BASE / "experiments/architecture_minus5/20261014/results",
                       -10: BASE / "experiments/resnet_lp4_minus10/20261015/results"}
    pairs, comparison_rows, predecessor_pairs, prediction_sets = [], [], [], 0
    for snr in exp.CONDITIONS:
        raw, meta = exp._target_cycles(snr, ordinary_root=Path(deploy["target_strong_root"]),
                        extreme_root=Path(deploy["target_extreme_root"]), export_root=None)
        assert meta == frozen and len(raw) == 88
        x = exp.data.raw_cycles(raw, meta)
        assert x.shape == (2474, 5, 4096) and np.isfinite(x).all()
        for (method, seed), (model, state) in models.items():
            obj = read(OUT / f"evaluations/{method}/seed{seed}/target_dev_snr{snr}.json")
            assert obj["truth"] == truth.tolist() and obj["source_only_selection"] == read(OUT / f"models/{method}/seed{seed}/run.json")["selected"]
            assert not obj["target_used_for_training_or_checkpoint_selection"] and not obj["final_holdout_evaluated"]
            predicted = exp.infer(model, x, torch.device("cuda"))
            assert np.array_equal(predicted, obj["predicted"]), (method, seed, snr)
            score = metric(truth, predicted)
            for key in score:
                close(score[key], obj["metrics"][key])
            assert all(torch.equal(v.detach().cpu(), state[k]) for k, v in model.state_dict().items())
            previous = read(reference_roots[snr] / f"evaluations/resnet_lp4/seed{seed}/target_dev_snr{snr}.json")
            assert previous["truth"] == truth.tolist()
            previous_metric = metric(truth, previous["predicted"])
            for key in previous_metric:
                close(previous_metric[key], previous["metrics"][key])
            pairs.append({"method": method, "seed": seed, "snr_db": snr,
                          "new_macro_f1_percent": 100 * score["macro_f1"],
                          "reference_resnet_lp4_macro_f1_percent": 100 * previous_metric["macro_f1"],
                          "new_minus_resnet_lp4_pp": 100 * (score["macro_f1"] - previous_metric["macro_f1"])})
            comparison_rows.append({"method": method, "seed": seed, "snr_db": snr,
                                    "macro_f1_percent": 100 * score["macro_f1"], "accuracy_percent": 100 * score["accuracy"]})
            predecessor = "a0" if method == "a1" else "b0"
            previous_proto = read(BASE / f"experiments/prototypes_a0_b0/c0b30c9/results/evaluations/{predecessor}/seed{seed}/target_dev_snr{snr}.json")
            assert previous_proto["truth"] == truth.tolist()
            predecessor_pairs.append({"method": method, "predecessor": predecessor, "seed": seed, "snr_db": snr,
                "new_macro_f1_percent": 100 * score["macro_f1"],
                "previous_macro_f1_percent": 100 * previous_proto["metrics"]["macro_f1"],
                "new_minus_previous_pp": 100 * (score["macro_f1"] - previous_proto["metrics"]["macro_f1"])})
            prediction_sets += 1
        for predecessor in ("a0", "b0"):
            for seed in exp.SEEDS:
                previous_proto = read(BASE / f"experiments/prototypes_a0_b0/c0b30c9/results/evaluations/{predecessor}/seed{seed}/target_dev_snr{snr}.json")
                assert previous_proto["truth"] == truth.tolist()
                previous_metric = metric(truth, previous_proto["predicted"])
                for key in previous_metric:
                    close(previous_metric[key], previous_proto["metrics"][key])
                comparison_rows.append({"method": predecessor, "seed": seed, "snr_db": snr,
                    "macro_f1_percent":100*previous_metric["macro_f1"], "accuracy_percent":100*previous_metric["accuracy"]})
        for seed in exp.SEEDS:
            old = read(reference_roots[snr] / f"evaluations/resnet_lp4/seed{seed}/target_dev_snr{snr}.json")["metrics"]
            comparison_rows.append({"method": "resnet_lp4", "seed": seed, "snr_db": snr,
                                    "macro_f1_percent": 100 * old["macro_f1"], "accuracy_percent": 100 * old["accuracy"]})
    summary = read(OUT / "reports/SUMMARY.json")["summary"]
    for row in summary:
        group = [r for r in comparison_rows if r["method"] == row["method"] and f"snr{r['snr_db']}" == row["condition"]]
        assert len(group) == 3
        for metric_name in ("macro_f1", "accuracy"):
            scores = np.asarray([r[metric_name + "_percent"] for r in group])
            close(scores.mean(), row[metric_name + "_mean"])
            close(scores.std(ddof=1), row[metric_name + "_sd"])
    report = OUT / "reports"
    exp.write_csv(report / "PAIRED_RESNET_LP4.csv", pairs)
    exp.write_csv(report / "COMPARISON_PER_SEED.csv", comparison_rows)
    exp.write_csv(report / "PAIRED_PREDECESSORS.csv", predecessor_pairs)
    combined = []
    for method in ("resnet_lp4", "a0", "a1", "b0", "b1"):
        for snr in exp.CONDITIONS:
            group = [r for r in comparison_rows if r["method"] == method and r["snr_db"] == snr]
            assert len(group) == 3
            values = np.asarray([r["macro_f1_percent"] for r in group])
            combined.append({"method":method, "snr_db":snr, "macro_f1_mean":float(values.mean()),
                             "macro_f1_sd":float(values.std(ddof=1))})
    exp.write_csv(report / "COMPARISON_SUMMARY.csv", combined)
    exp.write_json(report / "COMPARISON_SUMMARY.json", {"summary":combined,
        "new_models_selection":"source F1 then noisy source CE on exact F1 ties",
        "historical_models_selection":"source F1 then earliest exact F1 tie",
        "historical_models_retrained":False})
    exp.write_csv(report / "SOURCE_SELECTED_CHECKPOINTS.csv", selections)
    exp.write_csv(report / "SOURCE_VALIDATION_REPLAY.csv", source_replay)
    # 把嵌套的五档验证值展开成列，避免CSV单元格出现不可复算的Python字典。
    loss_rows = []
    for row in history_rows:
        loss_rows.append({"method": row["method"], "seed": row["seed"], "epoch": row["epoch"],
                          "train_loss": row["train_loss_mean"], "classification_ce": row.get("classification_ce_mean", ""),
                          "recovery_loss": row.get("source_restore_loss_mean", ""),
                          "source_clean_f1_percent": 100 * row["clean_f1"],
                          **{f"source_snr{k}_f1_percent": 100 * row["noisy_f1"][str(k)] for k in exp.VAL_NOISE}})
    exp.write_csv(report / "TRAIN_LOSS_HISTORY.csv", loss_rows)
    exp.write_csv(report / "B1_PRETRAIN_RECOVERY_HISTORY.csv", warmup_rows)
    exp.write_json(report / "WEIGHTS_SHA256.json", weights)
    file_rows = []
    for domain, metadata in (("source_train", tm), ("source_val", vm), ("target_dev", frozen)):
        for rid in sorted({r["record_id"] for r in metadata}):
            subset = [r for r in metadata if r["record_id"] == rid]
            file_rows.append({"domain": domain, "record_id": rid, "cycles": len(subset),
                              "healthy_cycles": sum(r["label"] == 0 for r in subset),
                              "fault_cycles": sum(r["label"] != 0 for r in subset)})
    exp.write_csv(report / "DATA_FILE_INDEX.csv", file_rows)
    assert all(sha(path) == expected for path, expected in protected.items())
    assert all(sha(path) == digest for path, digest in weights.items())
    exp.write_json(report / "VERIFICATION.json", {"status": "passed", "training_runs": 6,
         "joint_training_epochs_checked": 300, "b1_pretrain_epochs_checked": 30,
         "source_selected_checkpoints_independently_verified": 6, "source_validation_metric_replays": len(source_replay),
         "target_prediction_sets_exactly_reproduced": prediction_sets,
         "target_metrics_independently_recomputed": 24, "summary_groups_independently_verified": 8,
         "all_model_parameters_and_buffers_unchanged_during_replay": True,
         "protected_old_files_sha_unchanged": len(protected), "new_weights_sha_unchanged": len(weights),
         "target_dev_records": 88, "target_dev_cycles_per_condition": 2474,
         "class_cycle_counts": np.bincount(truth, minlength=6).tolist(), "unit_tests_passed": 64,
         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
         "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
         "final_holdout_evaluated": False, "target_checkpoint_selection": False,
         "elapsed_seconds": time.perf_counter() - started})
    files = {p.name: p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".log", ".txt")}
    files.update({p.relative_to(ROOT).as_posix(): p for folder in (ROOT / "logs", OUT)
                  for p in folder.rglob("*") if p.is_file() and p.suffix not in (".pt", ".pth")})
    files.update({"code/" + name: CODE / name for name in deploy["code_sha256"]})
    assert not any(p.suffix in (".mat", ".pt", ".pth", ".npz") for p in files.values())
    exp.write_json(ROOT / "RESULTS_EXPORT_MANIFEST.json", {"files": {name: {"sha256": sha(p), "bytes": p.stat().st_size}
                    for name, p in sorted(files.items())}, "raw_waveforms_included": False,
                    "model_weights_included": False, "environment_included": False})
    files["RESULTS_EXPORT_MANIFEST.json"] = ROOT / "RESULTS_EXPORT_MANIFEST.json"
    with tarfile.open(ROOT / "a1_b1_records.tar.gz", "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name)
    print(json.dumps(read(report / "VERIFICATION.json"), ensure_ascii=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("A1_B1_ALL_VERIFIED_AND_EXPORTED", flush=True)


if __name__ == "__main__":
    main()
