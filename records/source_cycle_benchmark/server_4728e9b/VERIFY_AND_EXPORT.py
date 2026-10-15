"""核验42组训练与504组冻结权重推理；只导出客观记录，不发布权重或原始MAT。"""
from pathlib import Path
import hashlib
import json
import os
import sys
import tarfile
import time

os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
CODE, OUT = ROOT / "code", ROOT / "results"
sys.path.insert(0, str(CODE / "scripts"))
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import source_cycle_benchmark as exp


def sha(path):
    """流式计算文件指纹，验证原输入、旧实验和新权重未改动。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    """读取实际JSON，所有六种子均必须存在。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual, expected, tolerance=1e-12):
    """指标只允许浮点舍入误差；类别预测另用逐元素完全相同检查。"""
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def metric(truth, predicted):
    """用sklearn独立复算六类指标，混淆矩阵行是真值、列是预测。"""
    cm = confusion_matrix(truth, predicted, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, predicted, labels=range(6), zero_division=0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "normal_false_alarms": int(cm[0, 1:].sum()), "normal_false_alarm_rate": float(cm[0, 1:].sum() / cm[0].sum()),
            "missed_fault_as_normal": int(cm[1:, 0].sum()), "confusion_matrix": cm.tolist()}


def csv(name, rows):
    """一次性写CSV；不同模型损失分项按字段并集展开，不丢失Xu或B1分项。"""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    exp.write_csv(OUT / "reports" / name, [{key: row.get(key, "") for key in fields} for row in rows])


def main():
    """完成源选模、预测重放、独立指标和汇总核验后才制作发布包。"""
    started = time.perf_counter()
    status, deploy, preflight = [read(ROOT / name) for name in ("RUN_STATUS.json", "DEPLOYMENT_MANIFEST.json", "PREFLIGHT.json")]
    assert status["status"] == "completed" and len(status["jobs"]) == 44
    protected = read(ROOT / "PROTECTED_FILES_SHA256.json")
    assert all(sha(path) == expected for path, expected in protected.items())
    assert all(sha(CODE / name) == expected for name, expected in deploy["code_sha256"].items())
    config = exp.configuration()
    views, provenance = exp.checked_source(Path(deploy["source_root"]), config)
    (_, tm, y, _), (_, vm, vy, val_clean, val_noise) = views
    source_inputs = {"clean": val_clean, **{f"snr{k}": v for k, v in val_noise.items()}}
    models, weights, selections, source_replay, history_rows, warmup_rows = {}, {}, [], [], [], []
    expected_steps = int(np.ceil(len(y) / config["batch_size"]))
    for method in exp.METHODS:
        for seed in exp.SEEDS:
            folder = OUT / f"models/{method}/seed{seed}"
            run, history = read(folder / "run.json"), read(folder / "history.json")
            assert (run["method"], run["seed"], run["protocol"]) == (method, seed, exp.PROTOCOL)
            assert run["config"] == config and run["source_split"] == provenance
            assert run["source_train_cycles"] == 962 and run["source_val_cycles"] == 221
            assert not run["target_used_for_training_or_selection"] and not run["final_holdout_evaluated"]
            assert run["initial_parameters_sha256"] == preflight["initial_parameters_sha256"][f"{method}/seed{seed}"]
            assert run["parameters"] == preflight["parameters"][method]
            assert run["first_training_noise_sha256"] == preflight["first_training_noise_sha256"][str(seed)]
            assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
            warmup = run["pretrain_history"]
            assert len(warmup) == (10 if method in ("b0", "b1") else 0)
            for epoch, loss in enumerate(warmup, 1):
                assert np.isfinite(loss) and loss >= 0
                warmup_rows.append({"method": method, "seed": seed, "epoch": epoch, "recovery_loss": loss})
            for row in history:
                assert np.isfinite(row["train_loss_mean"]) and row["optimizer_steps"] == expected_steps
                assert set(row["noisy_f1"]) == {"0", "5", "10", "20", "30"}
                close(row["noise_mean_f1"], np.mean(list(row["noisy_f1"].values())))
                close(row["noise_mean_ce"], np.mean(list(row["noisy_ce"].values())))
                parts = row["train_components"]
                expected = parts["ce"]
                if method == "xu_full":
                    expected += parts["recon"] + parts["dml"]
                    close(parts["dml"], parts["intra"] - parts["inter"], 1e-5)
                elif method == "b0":
                    expected += 0.2 * parts["rec"]
                elif method == "b1":
                    expected += parts["restore"]
                    close(parts["restore"], parts["complex"] + 0.4 * parts["phase"] + 0.1 * parts["noise"], 1e-5)
                close(row["train_loss_mean"], expected, 1e-4)
                history_rows.append({"method": method, "seed": seed, "epoch": row["epoch"],
                    "train_loss_batch_mean": row["train_loss_mean"], "train_loss_sample_weighted": row["train_loss_sample_weighted"],
                    "clean_f1_percent": 100 * row["clean_f1"], "clean_ce": row["clean_ce"],
                    **{f"snr{k}_f1_percent": 100 * row["noisy_f1"][str(k)] for k in config["val_noise_db"]},
                    **{f"snr{k}_ce": row["noisy_ce"][str(k)] for k in config["val_noise_db"]},
                    **{f"loss_{k}": value for k, value in parts.items()}})
            peak = max(r["clean_f1"] for r in history)
            eligible = [r for r in history if r["clean_f1"] >= peak - 0.01 - 1e-12]
            chosen = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["noise_mean_ce"], r["epoch"]))[0]
            assert chosen == run["selected"]
            saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            final = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
            assert saved["selected"] == chosen and saved["source_split"] == final["source_split"] == provenance
            assert saved["config"] == final["config"] == config and final["epoch"] == 50
            for state, epoch in ((saved["state"], chosen["epoch"]), (final["state"], 50)):
                assert all(int(value) == epoch * expected_steps for name, value in state.items() if name.endswith("num_batches_tracked"))
            exp.fix_randomness(seed)
            model = exp.build_model(method).cuda().eval().requires_grad_(False)
            model.load_state_dict(saved["state"], strict=True)
            for condition, x in source_inputs.items():
                predicted, ce = exp.predict(model, x, torch.device("cuda"), vy)
                score = metric(vy, predicted)
                expected_f1 = chosen["clean_f1"] if condition == "clean" else chosen["noisy_f1"][condition[3:]]
                expected_ce = chosen["clean_ce"] if condition == "clean" else chosen["noisy_ce"][condition[3:]]
                close(score["macro_f1"], expected_f1)
                close(ce, expected_ce, 1e-7)
                source_replay.append({"method": method, "seed": seed, "condition": condition,
                                      "macro_f1_percent": 100 * score["macro_f1"], "cross_entropy": ce})
            fully_saturated = [r["epoch"] for r in history if r["clean_f1"] == 1 and all(v == 1 for v in r["noisy_f1"].values())]
            old = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
            selections.append({"method": method, "seed": seed, "selected_epoch": chosen["epoch"],
                "f1_earliest_rule_epoch_diagnostic_only": old["epoch"], "eligible_epochs": len(eligible),
                "first_six_conditions_100_epoch": min(fully_saturated) if fully_saturated else "",
                "six_conditions_100_epochs": len(fully_saturated), "source_clean_f1_percent": 100 * chosen["clean_f1"],
                "source_five_noise_mean_f1_percent": 100 * chosen["noise_mean_f1"], "source_noise_mean_ce": chosen["noise_mean_ce"]})
            models[method, seed] = (model, saved["state"])
            for name in ("best.pt", "last.pt"):
                weights[str(folder / name)] = sha(folder / name)
    target_replays, score_rows = 0, []
    common_truth = None
    for snr in exp.CONDITIONS:
        name = "clean" if snr is None else f"snr{snr}"
        x, metadata = exp.target_cycles(snr, Path(deploy["snr_root"]), Path(deploy["strong_root"]), Path(deploy["extreme_root"]))
        truth = np.asarray([row["label"] for row in metadata], dtype=np.int64)
        if common_truth is None:
            common_truth = truth
        assert np.array_equal(truth, common_truth)
        assert np.bincount(truth, minlength=6).tolist() == [1106, 307, 297, 298, 237, 229]
        for (method, seed), (model, _) in models.items():
            obj = read(OUT / f"evaluations/{method}/seed{seed}/target_dev_{name}.json")
            assert obj["source_split"] == provenance and obj["truth"] == truth.tolist()
            assert not obj["final_holdout_evaluated"] and not obj["target_used_for_training_or_selection"]
            predicted, _ = exp.predict(model, x, torch.device("cuda"))
            assert np.array_equal(predicted, obj["predicted"]), (method, seed, name)
            actual = metric(truth, predicted)
            for key, value in actual.items():
                close(value, obj["metrics"][key])
            score_rows.append({"method": method, "seed": seed, "condition": name,
                               "macro_f1_percent": 100 * actual["macro_f1"], "accuracy_percent": 100 * actual["accuracy"]})
            target_replays += 1
        print("VERIFIED", name, "42 prediction sets", flush=True)
    summary = read(OUT / "reports/SUMMARY.json")["summary"]
    for row in summary:
        group = [r for r in score_rows if r["method"] == row["method"] and r["condition"] == row["condition"]]
        assert len(group) == 6
        for key in ("macro_f1", "accuracy"):
            values = np.asarray([r[key + "_percent"] for r in group])
            close(values.mean(), row[key + "_mean"])
            close(values.std(ddof=1), row[key + "_sd"])
    paired = []
    for row in score_rows:
        if row["method"] == "b1":
            for baseline in [m for m in exp.METHODS if m != "b1"]:
                other = next(r for r in score_rows if r["method"] == baseline and r["seed"] == row["seed"] and r["condition"] == row["condition"])
                paired.append({"baseline": baseline, "seed": row["seed"], "condition": row["condition"],
                               "b1_minus_baseline_macro_f1_pp": row["macro_f1_percent"] - other["macro_f1_percent"]})
    csv("SOURCE_SELECTED_CHECKPOINTS.csv", selections)
    csv("SOURCE_VALIDATION_REPLAY.csv", source_replay)
    csv("TRAINING_HISTORY.csv", history_rows)
    csv("PRETRAIN_HISTORY.csv", warmup_rows)
    csv("PAIRED_B1_VS_BASELINES.csv", paired)
    exp.write(OUT / "reports/WEIGHTS_SHA256.json", weights)
    assert all(sha(path) == value for path, value in protected.items())
    assert all(sha(path) == value for path, value in weights.items())
    for model, state in models.values():
        assert all(torch.equal(value.detach().cpu(), state[key]) for key, value in model.state_dict().items())
    verification = {"status": "passed", "training_runs_checked": 42, "classification_epochs_checked": len(history_rows),
        "recovery_pretrain_epochs_checked": len(warmup_rows), "source_validation_replays": len(source_replay),
        "target_prediction_sets_reproduced_exactly": target_replays, "summary_groups_independently_verified": len(summary),
        "protected_files_sha_unchanged": len(protected), "new_weights_sha_unchanged": len(weights),
        "source_split_manifest_sha256": provenance["manifest_sha256"], "unit_tests_passed": preflight["unit_tests_passed"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
        "final_holdout_evaluated": False, "target_training_or_checkpoint_selection": False,
        "elapsed_seconds": time.perf_counter() - started}
    exp.write(OUT / "reports/VERIFICATION.json", verification)
    files = {p.name: p for p in ROOT.iterdir() if p.is_file() and p.suffix in (".json", ".txt", ".log", ".py")}
    files.update({p.relative_to(ROOT).as_posix(): p for folder in (OUT, ROOT / "logs") for p in folder.rglob("*")
                  if p.is_file() and p.suffix not in (".pt", ".pth")})
    files.update({"code/" + name: CODE / name for name in deploy["code_sha256"]})
    assert not any(p.suffix in (".mat", ".npz", ".pt", ".pth", ".xlsx") for p in files.values())
    exp.write(ROOT / "RESULTS_EXPORT_MANIFEST.json", {"files": {name: {"sha256": sha(p), "bytes": p.stat().st_size}
              for name, p in sorted(files.items())}, "raw_waveforms_included": False, "model_weights_included": False,
              "environment_included": False})
    files["RESULTS_EXPORT_MANIFEST.json"] = ROOT / "RESULTS_EXPORT_MANIFEST.json"
    with tarfile.open(ROOT / "seven_models_six_seeds_records.tar.gz", "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name)
    print(json.dumps(verification, ensure_ascii=False), flush=True)
    print("ALL_42_RUNS_AND_504_PREDICTIONS_VERIFIED_AND_EXPORTED", flush=True)


if __name__ == "__main__":
    main()
