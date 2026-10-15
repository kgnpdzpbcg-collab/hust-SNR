"""LP4＋ResNet的−10 dB独立评估：复用源域选出的42/43/44三份权重。

输入为原架构实验快照、−5 dB记录和新生成的−10 dB目录。仅重放88条
实测Dev，预处理、切窗、CUDA精度均复用原版；不训练、不重新选择权重。
输出保留逐周期预测、六类指标、跨噪声档位比较及原文件指纹核验。
"""
import argparse
import hashlib
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
import torch

from evaluate_arch_suite_minus5 import read, sha, independent_metrics, check_metrics


def summarize(exp, output, results, config):
    """全部种子都纳入均值及样本标准差；下降值以百分点计，不做选种子。"""
    conditions = config["reference_target_conditions"] + [-10]
    rows, classes, summary, paired = [], [], [], []
    for seed in config["seeds"]:
        for condition in conditions:
            score = results[seed, condition]["metrics"]
            rows.append({"method": config["method"], "seed": seed, "condition": condition,
                         "macro_f1_percent": score["macro_f1"] * 100,
                         "accuracy_percent": score["accuracy"] * 100,
                         "fault_as_healthy": score["missed_fault_as_normal"],
                         "healthy_as_fault": score["normal_false_alarms"]})
            for label in range(6):
                classes.append({"seed": seed, "condition": condition, "label": label,
                                "f1_percent": score["per_class_f1"][label] * 100,
                                "recall_percent": score["per_class_recall"][label] * 100,
                                "fault_as_healthy": score["confusion_matrix"][label][0] if label else 0})
        for reference in config["reference_target_conditions"]:
            paired.append({"seed": seed, "reference_condition": reference,
                           "macro_f1_drop_pp": 100 * (results[seed, reference]["metrics"]["macro_f1"] -
                                                     results[seed, -10]["metrics"]["macro_f1"])})
    for condition in conditions:
        group = [row for row in rows if row["condition"] == condition]
        item = {"condition": condition, "n_seeds": len(group)}
        for metric in ("macro_f1", "accuracy"):
            values = np.asarray([row[metric + "_percent"] for row in group])
            item[metric + "_mean_percent"] = float(values.mean())
            item[metric + "_sd_percent"] = float(values.std(ddof=1))
        summary.append(item)
    report = output / "reports"
    exp.write_csv(report / "PER_SEED_RESULTS.csv", rows)
    exp.write_csv(report / "PER_CLASS_PER_SEED.csv", classes)
    exp.write_csv(report / "SUMMARY.csv", summary)
    exp.write_csv(report / "PAIRED_DEGRADATION.csv", paired)
    exp.write_json(report / "SUMMARY.json", {"protocol": config["protocol"], "summary": summary,
                   "seeds": config["seeds"], "method": config["method"], "final_holdout_evaluated": False})


def evaluate(args):
    """检查来源、88个输入及固定权重，先复现−5 dB，再两次独立推理−10 dB。"""
    started = time.perf_counter()
    config = read(args.config)
    if args.output.exists() or args.output.resolve().is_relative_to(args.reference_results.resolve()):
        raise ValueError("输出必须为不存在的独立目录，不能覆盖旧结果")
    sys.path.insert(0, str(args.reference_code / "scripts"))
    import hust_arch_suite_experiment as exp
    from prepare import OUT as PREPARE_OUT, ROOT as PREPARE_ROOT
    assert config["method"] == "resnet_lp4" and config["seeds"] == list(exp.SEEDS)
    assert config["reference_training_protocol"] == exp.PROTOCOL
    assert config["new_target_conditions_db"] == [-10]
    exp.configuration()
    assert torch.cuda.is_available(), "必须复用服务器原CUDA环境"
    frozen_path = PREPARE_OUT / "target_dev_clean_windows.json"
    frozen = read(frozen_path)
    ids = {row["record_id"] for row in frozen}
    truth = np.asarray([row["label"] for row in frozen], dtype=np.int64)
    assert len(ids) == config["target_dev_records"] == 88
    assert len(frozen) == config["target_dev_cycles"] == 2474
    assert np.bincount(truth, minlength=6).tolist() == [1106, 307, 297, 298, 237, 229]
    generation = read(args.minus10_root / "generated/GENERATION_MANIFEST.json")
    files = [row for row in generation["files"] if row["domain"] == "experimental" and row["partition"] == "dev"]
    assert len(files) == 88 and {row["record_id"] for row in files} == ids
    assert all(row["snr_db"] == -10 for row in files)
    # 保护历史快照、模型和预测；保留集只出现在生成清单中，不进入读取器。
    protected = {str(path): sha(path) for folder in
                 (args.reference_results, args.reference_code / "scripts", args.reference_code / "configs",
                  args.minus5_results) for path in folder.rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts}
    protected[str(frozen_path)] = sha(frozen_path)
    max_snr_error = 0.0
    for row in files:
        path = args.minus10_root / "generated" / row["file"]
        clean_path = PREPARE_ROOT / "data/target_dev/clean" / f"{row['record_id']}.mat"
        assert sha(path) == row["file_sha256"]
        protected[str(path)], protected[str(clean_path)] = sha(path), sha(clean_path)
        noisy, clean = loadmat(path)["I_noisy"], loadmat(clean_path)["name"]
        assert noisy.shape == clean.shape == (10031, 7)
        assert noisy.dtype == np.float64 and np.isfinite(noisy).all()
        assert np.array_equal(noisy[:, [0, -1]], clean[:, [0, -1]])
        measured = 10 * np.log10(np.mean(clean[:, 1:6]**2, axis=0) /
                                 np.mean((noisy[:, 1:6] - clean[:, 1:6])**2, axis=0))
        max_snr_error = max(max_snr_error, float(np.max(np.abs(measured + 10))))
    assert max_snr_error < 1e-6
    arrays = {}
    for snr, root in ((-5, args.minus5_root), (-10, args.minus10_root)):
        raw, metadata = exp.data.target_dev_from_mat(snr, root / "generated/target")
        assert metadata == frozen and set(raw) == ids
        arrays[snr] = exp.data.raw_cycles(raw, metadata)
        assert arrays[snr].shape == (2474, 5, 4096) and arrays[snr].dtype == np.float32
        assert np.isfinite(arrays[snr]).all()
    expected_weights = read(args.reference_results / "reports/WEIGHTS_SHA256.json")
    results, weights, selected = {}, {}, []
    method, device = config["method"], torch.device("cuda")
    args.output.mkdir(parents=True)
    for seed in config["seeds"]:
        folder = args.reference_results / f"models/{method}/seed{seed}"
        weight_path = folder / "best.pt"
        assert sha(weight_path) == expected_weights[str(weight_path)]
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        run = read(folder / "run.json")
        assert (checkpoint["method"], checkpoint["seed"], checkpoint["protocol"]) == (method, seed, exp.PROTOCOL)
        assert checkpoint["selected"] == run["selected"] and checkpoint["source_only"]
        assert not run["target_used_for_training_or_selection"] and not checkpoint["final_holdout_evaluated"]
        exp.fix_randomness(seed)
        model = exp.HUSTArchitectureCandidate(method).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["state"], strict=True)
        assert exp.parameter_count(model) == run["parameters"] == 241702
        for condition in config["reference_target_conditions"]:
            root = args.minus5_results if condition == -5 else args.reference_results
            name = "clean" if condition == "clean" else f"snr{condition}"
            old = read(root / f"evaluations/{method}/seed{seed}/target_dev_{name}.json")
            assert old["truth"] == truth.tolist() and old["selected_source"] == checkpoint["selected"]
            check_metrics(independent_metrics(truth, old["predicted"]), old["metrics"])
            results[seed, condition] = old
        previous = exp.infer(model, arrays[-5], device, batch=config["inference_batch_size"])
        assert np.array_equal(previous, results[seed, -5]["predicted"]), (seed, "历史−5 dB预测未复现")
        predicted = exp.infer(model, arrays[-10], device, batch=config["inference_batch_size"])
        replay = exp.infer(model, arrays[-10], device, batch=config["inference_batch_size"])
        assert np.array_equal(predicted, replay), (seed, "−10 dB两次推理不一致")
        score = exp.metrics(truth, predicted)
        check_metrics(independent_metrics(truth, predicted), score)
        assert all(torch.equal(value.detach().cpu(), checkpoint["state"][name]) for name, value in model.state_dict().items())
        result = {"method": method, "seed": seed, "condition": "snr-10", "protocol": config["protocol"],
                  "reference_training_protocol": exp.PROTOCOL, "record_families": 88, "windows": 2474,
                  "truth": truth.tolist(), "predicted": predicted.tolist(), "metrics": score,
                  "selected_source": checkpoint["selected"], "checkpoint_sha256": sha(weight_path),
                  "source_only_selection": True, "synthetic_target_noise": True,
                  "models_retrained": False, "final_holdout_evaluated": False}
        exp.write_json(args.output / f"evaluations/{method}/seed{seed}/target_dev_snr-10.json", result)
        results[seed, -10], weights[str(weight_path)] = result, result["checkpoint_sha256"]
        selected.append({"seed": seed, "selected_original_epoch": checkpoint["selected"]["epoch"],
                         "checkpoint_sha256": result["checkpoint_sha256"]})
        print(f"DEV resnet_lp4 seed{seed}: -10dB F1={score['macro_f1']*100:.4f}% Acc={score['accuracy']*100:.4f}%", flush=True)
        del model
    summarize(exp, args.output, results, config)
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    report = args.output / "reports"
    exp.write_json(report / "PROTECTED_FILES_SHA256.json", protected)
    exp.write_json(report / "WEIGHTS_SHA256.json", weights)
    exp.write_csv(report / "SOURCE_SELECTED_CHECKPOINTS.csv", selected)
    exp.write_json(report / "VERIFICATION.json", {"status": "passed", "fixed_checkpoint_count": 3,
                   "new_minus10_evaluation_sets": 3, "minus10_exact_prediction_replays": 3,
                   "historical_minus5_predictions_exactly_reproduced": 3,
                   "historical_metric_sets_independently_verified": 15, "new_metric_sets_independently_verified": 3,
                   "target_dev_records": 88, "target_dev_cycles": 2474,
                   "class_cycle_counts": np.bincount(truth, minlength=6).tolist(),
                   "max_original_record_phase_snr_error_db": max_snr_error,
                   "protected_files_sha_unchanged": len(protected),
                   "all_model_parameters_and_buffers_unchanged": True,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "models_retrained": 0, "source_data_loaded_in_evaluation": False,
                   "checkpoint_selection_changed": False, "final_holdout_evaluated": False,
                   "elapsed_seconds": time.perf_counter() - started,
                   "evaluation_script_sha256": sha(Path(__file__)), "configuration_sha256": sha(args.config),
                   "reference_frozen_metadata_sha256": sha(frozen_path),
                   "raw_minus10_cycle_array_sha256": hashlib.sha256(arrays[-10].tobytes()).hexdigest()})
    print("COMPLETE: 3 fixed weights evaluated; historical and repeated predictions exact", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("reference-code", "reference-results", "minus5-root", "minus5-results", "minus10-root", "output", "config"):
        parser.add_argument("--" + option, type=Path, required=True)
    evaluate(parser.parse_args())
