"""独立复算V4指标/源域选模，并重载十二个权重复现96组原MAT预测。

只读取本轮结果和冻结输入；验证输出独立保存，不覆盖原训练或测评。
原V1/Gate0文件按训练前指纹复核，原始电流和权重不加入发布文件。
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/scripts"))
import v2_experiment as v2
import torch
from v1_noise_consistency import fix_randomness
from v4_hpasf import HPASFCNN, counts_by_module, INITIAL_GAIN4


def read(path):
    """UTF-8解析现有JSON，不改写被核对的文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    """逐块计算原字节哈希，包含最佳权重和原V1文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    """独立计算的双精度指标只允许末位舍入误差。"""
    assert np.allclose(a, b, atol=1e-12, rtol=0), (a, b)


def main():
    """先检查训练/选模，再按条件回读MAT，逐条复现预测及全部指标。"""
    out = ROOT / "results"
    assert not (out / "VERIFICATION.json").exists()
    assert read(ROOT / "RUN_STATUS.json")["status"] == "completed"
    manifest = read(ROOT / "DEPLOYMENT_MANIFEST.json")
    preflight = read(ROOT / "PREFLIGHT.json")
    base = ROOT.parents[2]
    original = read(ROOT / "ORIGINAL_FILES_SHA256.json")
    for name, expected in original.items():
        assert sha(base / name) == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / "code" / name) == expected, name
    methods, seeds = manifest["methods"], manifest["seeds"]
    models, weights, checkpoints, selections, filter_stats = {}, {}, {}, {}, {}
    for method in methods:
        for seed in seeds:
            key = f"{method}/seed{seed}"
            folder = out / "models" / method / f"seed{seed}"
            info = read(folder / "run.json")
            history = read(folder / "history.json")
            assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
            assert all(np.isfinite(r["train_loss_mean"]) for r in history)
            assert info["epochs"] == 50 and info["batch_size"] == 32 and info["lr"] == .001
            assert not info["target_used_for_training_or_selection"] and not info["final_holdout_evaluated"]
            assert info["initial_cnn_sha256"] == preflight["initial_cnn_sha256"][str(seed)][method]
            # 独立展开源域选模规则，不调用被验证的score_checkpoints函数。
            maximum_clean = max(r["clean_f1"] for r in history)
            eligible = [r for r in history if r["clean_f1"] >= maximum_clean - .01 - 1e-12]
            chosen = sorted(eligible, key=lambda r: (-r["noise_mean_f1"], -r["clean_f1"], r["epoch"]))[0]
            assert chosen == info["selected"]
            path = folder / "best.pt"
            weights[key] = sha(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert checkpoint["source_only"] and checkpoint["mode"] == method and checkpoint["seed"] == seed
            assert checkpoint["selected"] == chosen and checkpoint["raw_points"] == 4096
            fix_randomness(seed)
            model = HPASFCNN(method).cuda()
            model.load_state_dict(checkpoint["state"], strict=True)
            model.eval()
            assert counts_by_module(model) == info["parameter_counts"]
            models[key] = model
            checkpoints[key] = checkpoint
            selections[key] = chosen
            # 两组可学习模型只检查唯一标量；门控逐样本效果另外记录。
            if method in ("static4", "hpasf"):
                logit = checkpoint["state"]["shaping.gate_logit"]
                filter_stats[key] = {"gate_logit": float(logit),
                                     "base_gain4": float(torch.sigmoid(logit)),
                                     "initial_gain4": INITIAL_GAIN4}
            for name, value in checkpoint["state"].items():
                if name.endswith("num_batches_tracked"):
                    assert int(value) == chosen["epoch"] * 24, (key, name)
    print("TRAINING_AND_SOURCE_SELECTION_VERIFIED: 12 models", flush=True)
    # 用最佳权重重算源Val的Clean及固定10/20/30dB，验证选中记录的分数。
    validation_raw, validation_meta, validation_labels = v2.source_data("val")
    validation_views = {"clean": v2.raw_cycles(validation_raw, validation_meta)}
    validation_views.update(v2.fixed_noisy_validation(validation_raw, validation_meta))
    for key, model in models.items():
        for condition, values in validation_views.items():
            predictions = v2.inference(model, values, torch.device("cuda"))
            score = f1_score(validation_labels, predictions, labels=list(range(6)), average="macro")
            expected = selections[key]["clean_f1"] if condition == "clean" else selections[key]["noisy_f1"][str(condition)]
            close(score, expected)
    print("SELECTED_SOURCE_VALIDATION_F1_REPRODUCED: 48 sets", flush=True)
    summary = read(out / "SUMMARY.json")
    recomputed = {}
    conditions = [None, 10, 15, 20, 25, 30, 35, 40]
    for snr in conditions:
        raw, metadata = v2.target_dev_from_mat(snr, ROOT / "target_snr")
        truth = np.array([r["label"] for r in metadata])
        assert len(metadata) == 2474 and len(raw) == 88
        inputs = v2.raw_cycles(raw, metadata)
        for method in methods:
            scores = []
            for seed in seeds:
                key = f"{method}/seed{seed}"
                condition = "clean" if snr is None else f"snr{snr}"
                result = read(out / "evaluations" / method / f"seed{seed}" / f"target_dev_{condition}.json")
                assert result["truth"] == truth.tolist()
                assert result["data_source"] == "original_MAT" and not result["final_holdout_evaluated"]
                assert result["source_only_selection"] == selections[key]
                predicted = np.array(result["predicted"])
                replay = v2.inference(models[key], inputs, torch.device("cuda"))
                assert np.array_equal(predicted, replay), (key, condition)
                cm = confusion_matrix(truth, predicted, labels=list(range(6)))
                metric = result["metrics"]
                close(f1_score(truth, predicted, labels=list(range(6)), average="macro"), metric["macro_f1"])
                close(accuracy_score(truth, predicted), metric["accuracy"])
                close(recall_score(truth, predicted, labels=list(range(6)), average="macro"), metric["balanced_accuracy"])
                close(f1_score(truth, predicted, labels=list(range(6)), average=None), metric["per_class_f1"])
                close(recall_score(truth, predicted, labels=list(range(6)), average=None), metric["per_class_recall"])
                assert cm.tolist() == metric["confusion_matrix"]
                false_alarm = int(cm[0].sum() - cm[0, 0])
                assert false_alarm == metric["normal_false_alarms"]
                close(false_alarm / cm[0].sum(), metric["normal_false_alarm_rate"])
                assert int(cm[:, 0].sum() - cm[0, 0]) == metric["missed_fault_as_normal"]
                scores.append(100 * metric["macro_f1"])
                recomputed[(method, seed, condition)] = scores[-1]
            row = next(r for r in summary["aggregates"] if r["method"] == method and r["condition"] == condition)
            close(np.mean(scores), row["macro_f1_mean_percent"])
            close(np.std(scores, ddof=1), row["macro_f1_sd_percent"])
            assert row["n_seeds"] == 3
        print("PREDICTIONS_AND_METRICS_VERIFIED:", condition, "12 models", flush=True)
    assert len(list((out / "evaluations").rglob("*.json"))) == 96
    for row in summary["paired_vs_lowpass3"]:
        method = row["method_minus_lowpass3"]
        diff = [recomputed[(method, s, row["condition"])] - recomputed[("lowpass3", s, row["condition"])] for s in seeds]
        close(diff, row["paired_deltas_pp"])
        close(np.mean(diff), row["mean_delta_pp"])
        assert sum(v > 0 for v in diff) == row["positive_seeds"]
    # 与已完成的V2低通控制组逐种子核对；记录差值，不强行认定必须相同。
    v2_parity = {}
    old = base / "experiments/v2_paifilter/4df46a37a6c2/results"
    for seed in seeds:
        old_history = read(old / "models/lowpass3" / f"seed{seed}/history.json")
        new_history = read(out / "models/lowpass3" / f"seed{seed}/history.json")
        deltas = {c: recomputed[("lowpass3", seed, c)] - 100 * read(old / "evaluations/lowpass3" / f"seed{seed}" / f"target_dev_{c}.json")["metrics"]["macro_f1"] for c in summary["conditions"]}
        v2_parity[str(seed)] = {"history_exactly_equal": old_history == new_history,
                               "target_macro_f1_difference_pp": deltas}
    for key, model in models.items():
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor.cpu(), checkpoints[key]["state"][name]), (key, name)
        assert sha(out / "models" / key / "best.pt") == weights[key]
    for name, expected in original.items():
        assert sha(base / name) == expected, name
    record = {"status": "passed", "code_commit": manifest["code_commit"],
              "trained_models": 12, "epochs_per_model": 50, "optimizer_updates_per_model": 1200,
              "prediction_vectors_exactly_reproduced": 96, "metric_sets_independently_recomputed": 96,
              "aggregate_rows_recomputed": 32, "source_only_checkpoint_selections_verified": 12,
              "selected_source_val_metric_sets_reproduced": 48,
              "original_files_unchanged": len(original), "weights_sha256": weights,
              "selected_checkpoints": selections, "fourth_harmonic_gate_statistics": filter_stats, "v2_lowpass3_parity": v2_parity,
              "in_memory_model_states_unchanged": True, "final_holdout_evaluated": False,
              "verification_script_sha256": sha(Path(__file__))}
    (out / "VERIFICATION.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("ALL_V4_VERIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
