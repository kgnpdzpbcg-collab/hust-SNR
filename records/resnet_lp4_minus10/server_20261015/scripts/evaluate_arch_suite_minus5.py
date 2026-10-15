"""九模型−5 dB独立压力测试：固定原源域选出的权重，只推理88条实测Dev。

从原训练代码快照导入模型、读取器和推理函数，保持预处理与CUDA精度。
先重放0 dB历史预测，再两次推理−5 dB；汇总三种子及逐类指标。
不训练、不微调、不重新选权重、不加载仿真或Final-Holdout波形。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch


def read(path):
    """读取冻结配置及历史结果，禁止在本入口改写旧文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """计算完整文件SHA，核对实际输入、权重与历史输出字节。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_metrics(truth, predicted):
    """由sklearn复算六类指标；混淆矩阵行真实、列预测，0为Healthy。"""
    matrix = confusion_matrix(truth, predicted, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=range(6), zero_division=0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(recall.mean()), "per_class_f1": f1.tolist(),
            "per_class_recall": recall.tolist(), "confusion_matrix": matrix.tolist(),
            "normal_false_alarms": int(matrix[0, 1:].sum()),
            "normal_false_alarm_rate": float(matrix[0, 1:].sum() / matrix[0].sum()),
            "missed_fault_as_normal": int(matrix[1:, 0].sum())}


def check_metrics(actual, expected):
    """逐项比较原指标实现与独立实现，只允许浮点末位误差。"""
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0)


def summarize(exp, output, results, config):
    """保留全部种子，统计样本标准差，并计算同种子0/5 dB到−5 dB的下降。"""
    rows, classes, summary, paired = [], [], [], []
    for method in config["methods"]:
        for seed in config["seeds"]:
            for snr in (5, 0, -5):
                result = results[method, seed, snr]
                score = result["metrics"]
                rows.append({"method": method, "seed": seed, "snr_db": snr,
                             "macro_f1_percent": score["macro_f1"] * 100,
                             "accuracy_percent": score["accuracy"] * 100,
                             "fault_as_healthy": score["missed_fault_as_normal"],
                             "healthy_as_fault": score["normal_false_alarms"]})
                for label in range(6):
                    classes.append({"method": method, "seed": seed, "snr_db": snr, "label": label,
                                    "f1_percent": score["per_class_f1"][label] * 100,
                                    "recall_percent": score["per_class_recall"][label] * 100,
                                    "fault_as_healthy": score["confusion_matrix"][label][0] if label else 0})
            for reference in (5, 0):
                paired.append({"method": method, "seed": seed, "reference_snr_db": reference,
                               "minus5_macro_f1_drop_pp": 100 * (results[method, seed, reference]["metrics"]["macro_f1"] -
                                                                      results[method, seed, -5]["metrics"]["macro_f1"])})
        for snr in (5, 0, -5):
            group = [row for row in rows if row["method"] == method and row["snr_db"] == snr]
            scores = np.asarray([row["macro_f1_percent"] for row in group])
            accuracies = np.asarray([row["accuracy_percent"] for row in group])
            summary.append({"method": method, "snr_db": snr, "n_seeds": len(group),
                            "macro_f1_mean_percent": float(scores.mean()), "macro_f1_sd_percent": float(scores.std(ddof=1)),
                            "accuracy_mean_percent": float(accuracies.mean()), "accuracy_sd_percent": float(accuracies.std(ddof=1))})
    report = output / "reports"
    exp.write_csv(report / "PER_SEED_RESULTS.csv", rows)
    exp.write_csv(report / "PER_CLASS_PER_SEED.csv", classes)
    exp.write_csv(report / "SUMMARY.csv", summary)
    exp.write_csv(report / "PAIRED_DEGRADATION.csv", paired)
    exp.write_json(report / "SUMMARY.json", {"protocol": config["protocol"], "summary": summary,
                   "seeds": config["seeds"], "methods": config["methods"], "final_holdout_evaluated": False})


def evaluate(args):
    """核对来源和数据后，只调用原推理函数；新记录写入独立输出目录。"""
    started = time.perf_counter()
    config = read(args.config)
    if args.output.exists() or args.output.resolve().is_relative_to(args.reference_results.resolve()):
        raise ValueError("输出必须是不存在的独立目录，不能写入旧实验结果目录")
    # 原模型和读取器均从已完成训练的快照导入，而非当前开发版本。
    sys.path.insert(0, str(args.reference_code / "scripts"))
    import hust_arch_suite_experiment as exp
    from prepare import OUT as PREPARE_OUT, ROOT as PREPARE_ROOT
    assert config["methods"] == list(exp.METHODS) and config["seeds"] == list(exp.SEEDS)
    assert config["reference_training_protocol"] == exp.PROTOCOL
    exp.configuration()
    assert torch.cuda.is_available(), "此任务使用服务器原CUDA环境"
    frozen_path = PREPARE_OUT / "target_dev_clean_windows.json"
    frozen = read(frozen_path)
    ids = {row["record_id"] for row in frozen}
    truth = np.asarray([row["label"] for row in frozen], dtype=np.int64)
    assert len(ids) == 88 and len(frozen) == 2474
    assert np.bincount(truth, minlength=6).tolist() == [1106, 307, 297, 298, 237, 229]
    generation = read(args.minus5_root / "generated/GENERATION_MANIFEST.json")
    dev_files = [row for row in generation["files"] if row["domain"] == "experimental" and row["partition"] == "dev"]
    assert {row["record_id"] for row in dev_files} == ids and len(dev_files) == 88
    assert all(row["snr_db"] == -5 for row in dev_files)
    # 保护旧模型、代码和结果；不读取保留集波形，仅加载原88条Dev。
    protected = {str(path): sha(path) for folder in (args.reference_results, args.reference_code / "scripts", args.reference_code / "configs")
                 for path in folder.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    protected[str(frozen_path)] = sha(frozen_path)
    expected_weights = read(args.reference_results / "reports/WEIGHTS_SHA256.json")
    max_snr_error = 0.0
    for row in dev_files:
        path = args.minus5_root / "generated" / row["file"]
        assert sha(path) == row["file_sha256"]
        clean_path = PREPARE_ROOT / "data/target_dev/clean" / f"{row['record_id']}.mat"
        protected[str(clean_path)] = sha(clean_path)
        protected[str(path)] = row["file_sha256"]
        noisy, clean = loadmat(path)["I_noisy"], loadmat(clean_path)["name"]
        assert noisy.shape == clean.shape == (10031, 7)
        assert noisy.dtype == np.float64 and np.isfinite(noisy).all()
        assert np.array_equal(noisy[:, [0, -1]], clean[:, [0, -1]])
        measured = 10 * np.log10(np.mean(clean[:, 1:6]**2, axis=0) /
                                 np.mean((noisy[:, 1:6] - clean[:, 1:6])**2, axis=0))
        max_snr_error = max(max_snr_error, float(np.max(np.abs(measured + 5))))
    assert max_snr_error < 1e-6
    raw, metadata = exp.data.target_dev_from_mat(-5, args.minus5_root / "generated/target")
    assert metadata == frozen and set(raw) == ids
    x_minus5 = exp.data.raw_cycles(raw, metadata)
    assert x_minus5.shape == (2474, 5, 4096) and x_minus5.dtype == np.float32
    assert np.isfinite(x_minus5).all()
    original_deploy = read(args.reference_results.parent / "DEPLOYMENT_MANIFEST.json")
    raw_zero, zero_meta = exp.target_raw(0, snr_root=None, strong_root=Path(original_deploy["target_strong_root"]), dev_export_root=None)
    assert zero_meta == frozen
    x_zero = exp.data.raw_cycles(raw_zero, zero_meta)
    results, weights, records = {}, {}, []
    device = torch.device("cuda")
    args.output.mkdir(parents=True)
    for method in exp.METHODS:
        for seed in exp.SEEDS:
            folder = args.reference_results / f"models/{method}/seed{seed}"
            weight_path = folder / "best.pt"
            assert sha(weight_path) == expected_weights[str(weight_path)]
            run = read(folder / "run.json")
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
            assert (checkpoint["method"], checkpoint["seed"], checkpoint["protocol"]) == (method, seed, exp.PROTOCOL)
            assert checkpoint["selected"] == run["selected"] and checkpoint["source_only"]
            assert not run["target_used_for_training_or_selection"] and not checkpoint["final_holdout_evaluated"]
            exp.fix_randomness(seed)
            model = exp.HUSTArchitectureCandidate(method).to(device).eval().requires_grad_(False)
            model.load_state_dict(checkpoint["state"], strict=True)
            assert exp.parameter_count(model) == run["parameters"]
            for reference in (5, 0):
                old = read(args.reference_results / f"evaluations/{method}/seed{seed}/target_dev_snr{reference}.json")
                assert old["truth"] == truth.tolist() and old["selected_source"] == checkpoint["selected"]
                check_metrics(independent_metrics(truth, old["predicted"]), old["metrics"])
                results[method, seed, reference] = old
            zero_pred = exp.infer(model, x_zero, device, batch=config["inference_batch_size"])
            assert np.array_equal(zero_pred, results[method, seed, 0]["predicted"]), (method, seed, "历史0 dB预测未复现")
            predicted = exp.infer(model, x_minus5, device, batch=config["inference_batch_size"])
            replay = exp.infer(model, x_minus5, device, batch=config["inference_batch_size"])
            assert np.array_equal(predicted, replay), (method, seed, "−5 dB两次推理不一致")
            score = exp.metrics(truth, predicted)
            check_metrics(independent_metrics(truth, predicted), score)
            assert all(torch.equal(value.detach().cpu(), checkpoint["state"][name]) for name, value in model.state_dict().items())
            result = {"method": method, "seed": seed, "condition": "snr-5", "protocol": config["protocol"],
                      "reference_training_protocol": exp.PROTOCOL, "record_families": 88, "windows": 2474,
                      "truth": truth.tolist(), "predicted": predicted.tolist(), "metrics": score,
                      "selected_source": checkpoint["selected"], "checkpoint_sha256": sha(weight_path),
                      "source_only_selection": True, "synthetic_target_noise": True,
                      "models_retrained": False, "final_holdout_evaluated": False}
            exp.write_json(args.output / f"evaluations/{method}/seed{seed}/target_dev_snr-5.json", result)
            results[method, seed, -5] = result
            weights[str(weight_path)] = result["checkpoint_sha256"]
            records.append({"method": method, "seed": seed, "selected_original_epoch": checkpoint["selected"]["epoch"],
                            "checkpoint_sha256": result["checkpoint_sha256"]})
            print(f"DEV {method} seed{seed}: -5dB F1={score['macro_f1']*100:.4f}% Acc={score['accuracy']*100:.4f}%", flush=True)
            del model
    summarize(exp, args.output, results, config)
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    report = args.output / "reports"
    exp.write_json(report / "PROTECTED_FILES_SHA256.json", protected)
    exp.write_json(report / "WEIGHTS_SHA256.json", weights)
    exp.write_csv(report / "SOURCE_SELECTED_CHECKPOINTS.csv", records)
    exp.write_json(report / "VERIFICATION.json", {"status": "passed", "fixed_checkpoint_count": 27,
                   "new_minus5_evaluation_sets": 27, "minus5_exact_prediction_replays": 27,
                   "historical_zero_db_prediction_sets_exactly_reproduced": 27,
                   "historical_reference_metric_sets_independently_verified": 54,
                   "new_metric_sets_independently_verified": 27, "target_dev_records": 88,
                   "target_dev_cycles": 2474, "class_cycle_counts": np.bincount(truth, minlength=6).tolist(),
                   "max_original_record_phase_snr_error_db": max_snr_error,
                   "old_files_and_minus5_dev_input_sha_unchanged": len(protected),
                   "all_model_parameters_and_buffers_unchanged": True,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "models_retrained": 0, "source_data_loaded": False, "checkpoint_selection_changed": False,
                   "final_holdout_evaluated": False, "elapsed_seconds": time.perf_counter() - started,
                   "evaluation_script_sha256": sha(Path(__file__)), "configuration_sha256": sha(args.config),
                   "reference_frozen_metadata_sha256": sha(frozen_path),
                   "raw_minus5_cycle_array_sha256": hashlib.sha256(x_minus5.tobytes()).hexdigest()})
    print("COMPLETE: 27 fixed checkpoints evaluated; all verification passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-code", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--minus5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/architecture_minus5_fixed_weights_v1.json")
    evaluate(parser.parse_args())
