"""4/−4/−8 dB独立压力实验：生成→独立核验→固定权重评价→预测重放。

复用4728e9b快照中的作者残差缩放、整周期读取器和原模型类。每阶段只在
明确的前阶段passed后继续；新MAT、预测、报告另存，原数据与权重只读。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from source_cycle_tsne import read, write, sha, save_csv


def setup(args):
    """核对配置及原代码快照，返回固定88记录的窗口元数据和数据路径。"""
    sys.path.insert(0, str(args.training_root / "code/scripts"))
    import source_cycle_benchmark as original
    from prepare import OUT
    config = read(args.config)
    assert config["noise_generation"]["levels_db"] == [4, -4, -8]
    assert config["models"] == ["resnet_lp4", "cnn1d_plain", "resnet1d", "b1"]
    assert config["seeds"] == [42, 44]
    deployment = read(args.training_root / "DEPLOYMENT_MANIFEST.json")
    assert deployment["code_commit"] == config["source_training_commit"]
    for name, expected in deployment["code_sha256"].items():
        assert sha(args.training_root / "code" / name) == expected, name
    assert read(args.training_root / "results/reports/VERIFICATION.json")["status"] == "passed"
    meta = read(OUT / "target_dev_clean_windows.json")
    ids = sorted({r["record_id"] for r in meta})
    assert len(meta) == 2474 and len(ids) == 88
    clean_root = args.training_root / "code/data/target_dev/clean"
    reference_root = Path(deployment["snr_root"]) / "SNR_10"
    return config, original, meta, ids, clean_root, reference_root


def matrix(path, field):
    """仅读取指定字段，严格保持作者10031×7的float64格式和有限值。"""
    import numpy as np
    from scipy.io import loadmat
    value = loadmat(path, variable_names=[field])[field]
    assert value.shape == (10031, 7) and value.dtype == np.float64 and np.isfinite(value).all(), path
    return value


def generate(args):
    """先核验全部88对输入，再用原缩放函数生成264份MAT，不重新抽噪声。"""
    import numpy as np
    from scipy.io import savemat
    config, _, _, ids, clean_root, reference_root = setup(args)
    from generate_strong_snr import scaled_author_current, phase_snr
    folder = args.output / "data"
    assert not folder.exists(), "拒绝覆盖已生成数据"
    inputs, pairs = {}, []
    for rid in ids:
        cp, rp = clean_root / f"{rid}.mat", reference_root / f"{rid}.mat"
        clean, reference = matrix(cp, "name"), matrix(rp, "I_noisy")
        assert np.array_equal(clean[:, [0, 6]], reference[:, [0, 6]])
        assert np.all(np.diff(clean[:, 0]) >= 0)
        assert np.max(np.abs(phase_snr(clean[:, 1:6], reference[:, 1:6]) - 10)) < 1e-6
        inputs[str(cp)], inputs[str(rp)] = sha(cp), sha(rp)
        pairs.append((rid, clean, reference))
    folder.mkdir(parents=True)
    files = []
    for level in config["noise_generation"]["levels_db"]:
        (folder / f"SNR_{level}").mkdir()
    for rid, clean, reference in pairs:
        for level in config["noise_generation"]["levels_db"]:
            value = clean.copy()
            value[:, 1:6] = scaled_author_current(clean[:, 1:6], reference[:, 1:6], level)
            p = folder / f"SNR_{level}/{rid}.mat"
            savemat(p, {"I_noisy": value}, do_compression=True)
            saved = matrix(p, "I_noisy")
            assert np.array_equal(saved, value) and np.array_equal(saved[:, [0, 6]], clean[:, [0, 6]])
            measured = phase_snr(clean[:, 1:6], saved[:, 1:6])
            assert np.max(np.abs(measured - level)) < 1e-6
            files.append({"file": str(p.relative_to(folder)), "record_id": rid, "snr_db": level,
                "noise_multiplier_from_author10": 10 ** ((10 - level) / 20),
                "measured_phase_snr_db": measured.tolist(), "file_sha256": sha(p),
                "matrix_sha256": hashlib.sha256(np.ascontiguousarray(saved).tobytes()).hexdigest()})
    assert len(files) == 264 and all(sha(p) == expected for p, expected in inputs.items())
    write(folder / "GENERATION_MANIFEST.json", {"status": "generated", "files": files,
        "input_sha256": inputs, "config": config, "old_inputs_unchanged": True,
        "independent_measurements": False, "final_holdout_accessed": False})
    print("GENERATED", len(files), "MAT; verify-data required before evaluation", flush=True)


def verify_data(args):
    """从保存的MAT独立重算能量比、噪声方向和缩放比例，再允许评价。

    不调用生成函数来验证自身。检查所有264份导出、176份原输入SHA、
    三档相同编号，以及原时间列和角速度列逐值不变。
    """
    import numpy as np
    config, _, _, ids, clean_root, reference_root = setup(args)
    folder = args.output / "data"
    manifest = read(folder / "GENERATION_MANIFEST.json")
    assert manifest["config"] == config
    assert all(sha(p) == expected for p, expected in manifest["input_sha256"].items())
    maximum, residual_error = 0.0, 0.0
    for level in config["noise_generation"]["levels_db"]:
        assert sorted(p.stem for p in (folder / f"SNR_{level}").glob("*.mat")) == ids
    for entry in manifest["files"]:
        path = folder / entry["file"]
        assert sha(path) == entry["file_sha256"]
        value = matrix(path, "I_noisy")
        clean = matrix(clean_root / f"{entry['record_id']}.mat", "name")
        reference = matrix(reference_root / f"{entry['record_id']}.mat", "I_noisy")
        assert np.array_equal(value[:, [0, 6]], clean[:, [0, 6]])
        noise, reference_noise = value[:, 1:6] - clean[:, 1:6], reference[:, 1:6] - clean[:, 1:6]
        measured = 10 * np.log10(np.mean(clean[:, 1:6] ** 2, axis=0) / np.mean(noise ** 2, axis=0))
        error = float(np.max(np.abs(measured - entry["snr_db"])))
        maximum = max(maximum, error)
        factor = 10 ** ((10 - entry["snr_db"]) / 20)
        difference = float(np.max(np.abs(noise - reference_noise * factor)))
        residual_error = max(residual_error, difference)
        assert np.allclose(noise, reference_noise * factor, rtol=1e-12, atol=1e-12)
    assert maximum < 1e-6
    write(folder / "VERIFICATION.json", {"status": "passed", "files_checked": 264,
        "input_sha256_checked": 176, "max_whole_record_per_phase_snr_error_db": maximum,
        "max_residual_scaling_absolute_error": residual_error, "time_speed_columns_unchanged": True,
        "new_gaussian_noise_added": False, "final_holdout_accessed": False})
    print("DATA_VERIFIED", maximum, "dB max SNR error", flush=True)


def evaluate(args, replay=False):
    """8份best在三档新数据上评价；重放时只核对，不改写正式预测。

    同一档位的2474个周期一次制作后供8模型使用。最佳epoch和全部state
    来自原训练，无优化器、梯度或源验证选模；输出六类F1/Recall和混淆矩阵。
    """
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
    config, original, metadata, _, _, _ = setup(args)
    from v1_noise_consistency import fix_randomness
    assert read(args.output / "data/VERIFICATION.json")["status"] == "passed"
    generation = read(args.output / "data/GENERATION_MANIFEST.json")
    assert all(sha(args.output / "data" / entry["file"]) == entry["file_sha256"]
               for entry in generation["files"])
    folder = args.output / "evaluation"
    if not replay:
        assert not folder.exists(), "拒绝覆盖既有评价"
        folder.mkdir()
    protected, models = {}, {}
    for method in config["models"]:
        for seed in config["seeds"]:
            model_folder = args.training_root / f"results/models/{method}/seed{seed}"
            for name in ("best.pt", "last.pt", "run.json", "history.json"):
                path = model_folder / name
                protected[str(path)] = sha(path)
            saved = torch.load(model_folder / "best.pt", map_location="cpu", weights_only=True)
            assert (saved["method"], saved["seed"], saved["protocol"]) == (method, seed, original.PROTOCOL)
            assert saved["source_split"]["manifest_sha256"] == config["source_split_manifest_sha256"]
            assert not saved["final_holdout_evaluated"]
            fix_randomness(seed)
            model = original.build_model(method).cuda().eval()
            model.load_state_dict(saved["state"], strict=True)
            models[method, seed] = model, saved["selected"]
    if replay:
        assert read(folder / "PROTECTED_FILES_SHA256.json") == protected
    rows, classes = [], []
    truth = np.asarray([r["label"] for r in metadata], dtype=np.int64)
    assert np.bincount(truth, minlength=6).tolist() == [1106, 307, 297, 298, 237, 229]
    for level in config["noise_generation"]["levels_db"]:
        raw, actual = original.data.target_dev_from_mat(level, args.output / "data")
        assert actual == metadata and len(raw) == 88
        x = original.data.raw_cycles(raw, metadata)
        for (method, seed), (model, selected) in models.items():
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            with torch.inference_mode():
                logits = np.concatenate([model(torch.from_numpy(x[i:i + 128]).cuda()).cpu().numpy()
                                         for i in range(0, len(x), 128)])
            predicted = logits.argmax(1)
            assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items())
            score = original.metrics(truth, predicted)
            assert np.isclose(score["macro_f1"], f1_score(truth, predicted, average="macro", labels=list(range(6))))
            assert np.isclose(score["accuracy"], accuracy_score(truth, predicted))
            cm = confusion_matrix(truth, predicted, labels=list(range(6)))
            condition = f"snr{level}"
            path = folder / f"evaluations/{method}/seed{seed}/target_dev_{condition}.json"
            if replay:
                prior = read(path)
                assert prior["truth"] == truth.tolist() and prior["predicted"] == predicted.tolist()
                assert prior["metrics"] == score
            else:
                write(path, {"method": method, "seed": seed, "condition": condition,
                    "protocol": config["protocol"], "truth": truth.tolist(), "predicted": predicted.tolist(),
                    "metrics": score, "record_families": 88, "windows": 2474,
                    "source_only_selection": selected, "weights_sha256": protected[str(args.training_root / f'results/models/{method}/seed{seed}/best.pt')],
                    "target_used_for_training_or_selection": False, "final_holdout_evaluated": False})
            rows.append({"method": method, "seed": seed, "condition": condition,
                "macro_f1_percent": 100 * score["macro_f1"], "accuracy_percent": 100 * score["accuracy"],
                "fault_as_healthy": int(cm[1:, 0].sum()), "healthy_as_fault": int(cm[0, 1:].sum())})
            precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(range(6)), zero_division=0)
            classes.extend({"method": method, "seed": seed, "condition": condition, "label": label,
                "f1_percent": 100 * f1[label], "recall_percent": 100 * recall[label], "support": int(support[label])} for label in range(6))
            print("REPLAYED" if replay else "EVALUATED", method, seed, level, "F1", 100 * score["macro_f1"], flush=True)
    assert len(rows) == 24 and all(sha(p) == value for p, value in protected.items())
    if replay:
        # 独立复算已发布CSV，防止汇总时混种子或列错档位。
        import csv
        with (folder / "PER_SEED_RESULTS.csv").open() as stream:
            published = list(csv.DictReader(stream))
        assert len(published) == len(rows)
        for row, prior in zip(rows, published):
            assert (row["method"], str(row["seed"]), row["condition"]) == (prior["method"], prior["seed"], prior["condition"])
            assert abs(row["macro_f1_percent"] - float(prior["macro_f1_percent"])) < 1e-10
        write(folder / "VERIFICATION.json", {"status": "passed", "prediction_sets_replayed": 24,
            "metrics_independently_recomputed": 24, "protected_weight_and_history_files_unchanged": len(protected),
            "retrained": False, "final_holdout_evaluated": False})
    else:
        save_csv(folder / "PER_SEED_RESULTS.csv", rows)
        save_csv(folder / "PER_CLASS_PER_SEED.csv", classes)
        write(folder / "PROTECTED_FILES_SHA256.json", protected)


def main():
    """四阶段显式调用；阶段失败保留现场，不自动重试或改变参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "verify-data", "evaluate", "verify-evaluation"))
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args)
    elif args.stage == "verify-data":
        verify_data(args)
    else:
        evaluate(args, replay=args.stage == "verify-evaluation")


if __name__ == "__main__":
    main()
