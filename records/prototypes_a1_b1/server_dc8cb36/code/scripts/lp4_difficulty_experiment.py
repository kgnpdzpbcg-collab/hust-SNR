"""LP4任务难度实验：三种45维分类器，另复用A′ CNN/ResNet固定权重。

train阶段只打开34/15仿真Train/Val，所有选择完成后才允许evaluate实测Dev。
原数据适配、噪声、划分及参考模型均从原架构实验代码快照导入。
缓存及新权重仅留服务器；报告保留全部种子、候选及逐类结果。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


def read(path):
    """读取冻结元数据及配置；不从文件名猜测故障类型。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """流式指纹记录数据、特征缓存和实际选中的分类器文件。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(values):
    """相同数据视图的数值指纹不受NPZ导出时间或压缩方式影响。"""
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def score(exp, truth, predicted):
    """以sklearn复核原六类指标，避免新的指标实现改变对照分数。"""
    result = exp.metrics(truth, predicted)
    matrix = confusion_matrix(truth, predicted, labels=range(6))
    _, recall, f1, _ = precision_recall_fscore_support(truth, predicted, labels=range(6), zero_division=0)
    np.testing.assert_allclose(result["per_class_f1"], f1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["per_class_recall"], recall, atol=1e-12, rtol=0)
    assert result["confusion_matrix"] == matrix.tolist()
    assert abs(result["accuracy"] - accuracy_score(truth, predicted)) < 1e-12
    return result


@torch.no_grad()
def mlp_predict(model, features, device):
    """仅对缩放后的45维向量推理，关闭梯度并固定eval状态。"""
    model.eval()
    return np.concatenate([model(torch.from_numpy(features[i:i + 128]).to(device)).argmax(1).cpu().numpy()
                           for i in range(0, len(features), 128)])


def validation_row(exp, vy, val, predict, epoch, **extra):
    """与原神经网络同规则：Clean及五档源噪声F1，绝不使用实测成绩。"""
    clean_f1 = score(exp, vy, predict(val["clean"]))["macro_f1"]
    noise = {str(level): score(exp, vy, predict(val[str(level)]))["macro_f1"] for level in (0, 5, 10, 20, 30)}
    return {"epoch": epoch, "clean_f1": clean_f1, "noisy_f1": noise,
            "noise_mean_f1": float(np.mean(list(noise.values()))), **extra}


def train(args, exp, config):
    """源数据独立训练三个分类器，先缓存共享50轮噪声，再做源域选模。"""
    from lp4_difficulty_models import extract_features, SmallHarmonicMLP, FEATURE_VERSION
    from prepare import ROOT, OUT as PREPARE_OUT
    started = time.perf_counter()
    seed = args.seed
    exp.fix_randomness(seed)
    device = torch.device("cuda")
    tr, tm, y = exp.data.source_data("train")
    vr, vm, vy = exp.data.source_data("val")
    assert (len(tr), len(vr), len(tm), len(vm)) == (34, 15, 738, 302)
    assert not set(tr) & set(vr)
    cache = args.output / "cache" / f"seed{seed}"
    cache.mkdir(parents=True, exist_ok=False)
    inputs = {str(ROOT / f"data/source/{rid}.mat"): sha(ROOT / f"data/source/{rid}.mat") for rid in (*tr, *vr)}
    for name in ("source_train_windows.json", "source_val_windows.json"):
        inputs[str(PREPARE_OUT / name)] = sha(PREPARE_OUT / name)
    for level in (0, 5):
        for rid in vr:
            path = args.source_strong_root / f"generated/source/SNR_{level}/{rid}.mat"
            inputs[str(path)] = sha(path)
    clean_raw = exp.data.raw_cycles(tr, tm)
    clean, max_error = extract_features(clean_raw, device, verify_reconstruction=True)
    val_clean_raw = exp.data.raw_cycles(vr, vm)
    val_clean, err = extract_features(val_clean_raw, device, verify_reconstruction=True)
    max_error = max(max_error, err)
    fixed = exp.data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5, 10, 20, 30),
                  materialized_root=args.source_strong_root / "generated/source")
    val = {"clean": val_clean}
    for level, raw in fixed.items():
        val[str(level)], err = extract_features(raw, device, verify_reconstruction=True)
        max_error = max(max_error, err)
    noise, schedule = [], []
    for epoch in range(1, 51):
        raw = exp.data.fresh_noisy_source(tr, tm, seed, epoch)
        features, err = extract_features(raw, device, verify_reconstruction=True)
        max_error = max(max_error, err)
        noise.append(features)
        schedule.append({"epoch": epoch, "snr_by_record": exp.data.training_noise_schedule(sorted(tr), seed, epoch)})
        if epoch % 10 == 0:
            print(f"FEATURES seed{seed}: {epoch}/50 shared noise epochs", flush=True)
    noise = np.stack(noise)
    dataset = np.concatenate((clean, noise.reshape(-1, 45)))
    labels = np.tile(y, 51)
    # 每条Clean权重1，每个噪声视图1/50，总贡献比1:1；缩放器同样加权拟合。
    weights = np.r_[np.ones(len(y)), np.full(len(y) * 50, 1 / 50)]
    assert dataset.shape == (37638, 45) and abs(weights[:738].sum() - weights[738:].sum()) < 1e-9
    scaler = StandardScaler().fit(dataset, sample_weight=weights)
    scaled = scaler.transform(dataset).astype(np.float32)
    clean_scaled, noise_scaled = scaled[:738], scaled[738:].reshape(50, 738, 45)
    val_scaled = {name: scaler.transform(values).astype(np.float32) for name, values in val.items()}
    np.savez_compressed(cache / "source_features.npz", clean=clean, noise=noise, labels=y, val_labels=vy, **{f"val_{k}": v for k, v in val.items()})
    source_replays, training_records = [], []
    for method in ("lp4_logreg", "lp4_rbf_svm"):
        rows, candidates = [], []
        for index, C in enumerate(config["logreg" if method == "lp4_logreg" else "svm"]["C"], 1):
            estimator = (LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-6)
                         if method == "lp4_logreg" else SVC(C=C, kernel="rbf", gamma="scale", cache_size=512))
            start_fit = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(scaled, labels, sample_weight=weights)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                raise RuntimeError(f"{method}/seed{seed}/C{C}未收敛：{convergence}")
            row = validation_row(exp, vy, val_scaled, estimator.predict, index,
                                 params={"C": C}, fit_seconds=time.perf_counter() - start_fit)
            rows.append(row)
            candidates.append(estimator)
            print(f"FIT {method} seed{seed} C={C}: srcClean={row['clean_f1']:.4f} srcFiveNoise={row['noise_mean_f1']:.4f}", flush=True)
        chosen = exp.score_checkpoints(rows)
        estimator = candidates[chosen["epoch"] - 1]
        folder = args.output / f"models/{method}/seed{seed}"
        folder.mkdir(parents=True, exist_ok=False)
        joblib.dump({"estimator": estimator, "scaler": scaler, "feature_version": FEATURE_VERSION,
                     "method": method, "seed": seed, "selected": chosen}, folder / "best.joblib")
        run = {"method": method, "seed": seed, "protocol": config["protocol"], "selected": chosen,
               "candidates": rows, "source_only": True, "target_used_for_training_or_selection": False,
               "source_training_views": 37638, "source_noise_epochs": 50,
               "feature_dimension": 45, "feature_version": FEATURE_VERSION,
               "scaler_mean": scaler.mean_.tolist(), "scaler_var": scaler.var_.tolist(),
               "support_vectors": len(estimator.support_) if method == "lp4_rbf_svm" else None,
               "source_features_sha256": sha(cache / "source_features.npz")}
        exp.write_json(folder / "run.json", run)
        training_records.append(run)
        for name, values in val_scaled.items():
            pred = estimator.predict(values)
            selected_f1 = chosen["clean_f1"] if name == "clean" else chosen["noisy_f1"][name]
            assert abs(score(exp, vy, pred)["macro_f1"] - selected_f1) < 1e-12
            source_replays.append({"method": method, "seed": seed, "condition": name,
                                   "truth": vy.tolist(), "predicted": pred.tolist(), "metrics": score(exp, vy, pred)})
    # 小MLP与原神经网络使用相同50轮、配对CE、样本顺序和Adam预算。
    exp.fix_randomness(seed)
    model = SmallHarmonicMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
    history, states = [], {}
    for epoch in range(1, 51):
        order = np.random.default_rng(seed + 901 + epoch).permutation(len(y))
        model.train()
        losses = []
        for start in range(0, len(order), 32):
            ix = order[start:start + 32]
            a = torch.from_numpy(clean_scaled[ix]).to(device)
            b = torch.from_numpy(noise_scaled[epoch - 1, ix]).to(device)
            yy = torch.from_numpy(y[ix]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.cat((a, b)))
            loss = .5 * torch.nn.functional.cross_entropy(logits[:len(ix)], yy) + .5 * torch.nn.functional.cross_entropy(logits[len(ix):], yy)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = validation_row(exp, vy, val_scaled, lambda x: mlp_predict(model, x, device), epoch, train_loss_mean=float(np.mean(losses)))
        history.append(row)
        states[epoch] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"MLP seed{seed} epoch{epoch}/50: srcClean={row['clean_f1']:.4f} srcFiveNoise={row['noise_mean_f1']:.4f}", flush=True)
    chosen = exp.score_checkpoints(history)
    model.load_state_dict(states[chosen["epoch"]], strict=True)
    folder = args.output / f"models/lp4_mlp/seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    torch.save({"method": "lp4_mlp", "seed": seed, "state": states[chosen["epoch"]], "selected": chosen,
                "feature_version": FEATURE_VERSION, "source_only": True}, folder / "best.pt")
    joblib.dump(scaler, folder / "scaler.joblib")
    exp.write_json(folder / "history.json", history)
    exp.write_json(folder / "run.json", {"method": "lp4_mlp", "seed": seed, "protocol": config["protocol"],
                   "selected": chosen, "parameters": 1670, "updates": 1200, "epochs": 50,
                   "source_only": True, "target_used_for_training_or_selection": False,
                   "scaler_mean": scaler.mean_.tolist(), "scaler_var": scaler.var_.tolist()})
    for name, values in val_scaled.items():
        predicted = mlp_predict(model, values, device)
        expected = chosen["clean_f1"] if name == "clean" else chosen["noisy_f1"][name]
        assert abs(score(exp, vy, predicted)["macro_f1"] - expected) < 1e-12
        source_replays.append({"method": "lp4_mlp", "seed": seed, "condition": name,
                               "truth": vy.tolist(), "predicted": predicted.tolist(), "metrics": score(exp, vy, predicted)})
    assert all(sha(Path(path)) == digest for path, digest in inputs.items())
    exp.write_json(args.output / f"reports/source_seed{seed}.json", {
                   "status": "completed_and_verified", "seed": seed, "source_train_records": 34, "source_val_records": 15,
                   "source_train_cycles": 738, "source_val_cycles": 302, "target_data_accessed": False,
                   "source_noise_schedule": schedule, "train_features_sha256": array_sha(dataset),
                   "scaler_fit_only_on_train": True, "weighted_train_mean": scaler.mean_.tolist(),
                   "max_lp4_md_128_reconstruction_error": max_error, "source_inputs_sha256": inputs,
                   "source_selected_replays": source_replays, "elapsed_seconds": time.perf_counter() - started})
    print(f"TRAIN_COMPLETE seed{seed}: three selected classifiers; target not accessed", flush=True)


def evaluate(args, exp, config):
    """所有源域选择冻结后，统一评估11种Dev条件与两个历史固定权重对照。"""
    from lp4_difficulty_models import extract_features, SmallHarmonicMLP
    from prepare import OUT as PREPARE_OUT
    from v4_hpasf import HPASFCNN
    started = time.perf_counter()
    frozen = read(PREPARE_OUT / "target_dev_clean_windows.json")
    truth = np.asarray([row["label"] for row in frozen], dtype=np.int64)
    assert len(frozen) == 2474 and len({row["record_id"] for row in frozen}) == 88
    assert np.bincount(truth, minlength=6).tolist() == [1106, 307, 297, 298, 237, 229]
    classifiers, controls, selected, protected = {}, {}, {}, {}
    device = torch.device("cuda")
    for seed in config["seeds"]:
        source = read(args.output / f"reports/source_seed{seed}.json")
        assert source["status"] == "completed_and_verified" and not source["target_data_accessed"]
        for method in config["new_methods"]:
            folder = args.output / f"models/{method}/seed{seed}"
            run = read(folder / "run.json")
            assert run["source_only"] and not run["target_used_for_training_or_selection"]
            selected[method, seed] = run["selected"]
            if method == "lp4_mlp":
                checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                scaler = joblib.load(folder / "scaler.joblib")
                model = SmallHarmonicMLP().to(device).eval().requires_grad_(False)
                model.load_state_dict(checkpoint["state"], strict=True)
                classifiers[method, seed] = model, scaler
            else:
                saved = joblib.load(folder / "best.joblib")
                classifiers[method, seed] = saved["estimator"], saved["scaler"]
            for path in folder.iterdir():
                protected[str(path)] = sha(path)
        for method, folder in (("lp4_aprime_cnn", args.reference_aprime / f"results/models/lowpass4/seed{seed}"),
                               ("lp4_resnet", args.reference_results / f"models/resnet_lp4/seed{seed}")):
            exp.fix_randomness(seed)
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            model = (HPASFCNN("lowpass4") if method == "lp4_aprime_cnn" else exp.HUSTArchitectureCandidate("resnet_lp4")).to(device).eval().requires_grad_(False)
            model.load_state_dict(checkpoint["state"], strict=True)
            selected[method, seed] = checkpoint["selected"]
            controls[method, seed] = model, checkpoint["state"]
            protected[str(folder / "best.pt")] = sha(folder / "best.pt")
    all_methods = config["new_methods"] + config["reference_methods"]
    rows, per_class, summary, checked, feature_errors = [], [], [], [], []
    for condition in config["target_conditions"]:
        level = None if condition == "clean" else condition
        name = "clean" if level is None else f"snr{level}"
        raw, metadata = (exp.data.target_dev_from_mat(-5, args.minus5_root / "generated/target") if level == -5 else
                         exp.target_raw(level, snr_root=args.target_snr_root, strong_root=args.target_strong_root, dev_export_root=None))
        assert metadata == frozen
        x = exp.data.raw_cycles(raw, metadata)
        features, error = extract_features(x, device, verify_reconstruction=True)
        feature_errors.append({"condition": name, "max_reconstruction_error": error, "feature_sha256": array_sha(features)})
        for method in all_methods:
            for seed in config["seeds"]:
                if method in config["new_methods"]:
                    model, scaler = classifiers[method, seed]
                    scaled = scaler.transform(features).astype(np.float32)
                    predicted = mlp_predict(model, scaled, device) if method == "lp4_mlp" else model.predict(scaled)
                    replay = mlp_predict(model, scaled, device) if method == "lp4_mlp" else model.predict(scaled)
                    assert np.array_equal(predicted, replay)
                else:
                    model, state = controls[method, seed]
                    exp.fix_randomness(seed)
                    predicted = exp.infer(model, x, device)
                    assert all(torch.equal(value.detach().cpu(), state[k]) for k, value in model.state_dict().items())
                    if level != -5:
                        old_root, old_method = (args.reference_aprime / "results", "lowpass4") if method == "lp4_aprime_cnn" else (args.reference_results, "resnet_lp4")
                        old = read(old_root / f"evaluations/{old_method}/seed{seed}/target_dev_{name}.json")
                        assert old["truth"] == truth.tolist() and np.array_equal(predicted, old["predicted"]), (method, seed, name)
                        checked.append({"method": method, "seed": seed, "condition": name})
                    elif method == "lp4_resnet":
                        old = read(args.reference_minus5_results / f"evaluations/resnet_lp4/seed{seed}/target_dev_snr-5.json")
                        assert np.array_equal(predicted, old["predicted"])
                        checked.append({"method": method, "seed": seed, "condition": name})
                metrics = score(exp, truth, predicted)
                exp.write_json(args.output / f"evaluations/{method}/seed{seed}/target_dev_{name}.json", {
                     "protocol": config["protocol"], "method": method, "seed": seed, "condition": name,
                     "record_families": 88, "windows": 2474, "truth": truth.tolist(), "predicted": predicted.tolist(),
                     "metrics": metrics, "selected_source": selected[method, seed], "source_only_selection": True,
                     "synthetic_target_noise": level in (0, 5, -5), "final_holdout_evaluated": False})
                rows.append({"method": method, "seed": seed, "condition": name, "macro_f1_percent": 100 * metrics["macro_f1"],
                             "accuracy_percent": 100 * metrics["accuracy"], "fault_as_healthy": metrics["missed_fault_as_normal"],
                             "healthy_as_fault": metrics["normal_false_alarms"]})
                for label in range(6):
                    per_class.append({"method": method, "seed": seed, "condition": name, "label": label,
                                      "f1_percent": 100 * metrics["per_class_f1"][label], "recall_percent": 100 * metrics["per_class_recall"][label]})
        print(f"TARGET_COMPLETE {name}: all15 classifiers", flush=True)
    for method in all_methods:
        for condition in config["target_conditions"]:
            name = "clean" if condition == "clean" else f"snr{condition}"
            group = [row for row in rows if row["method"] == method and row["condition"] == name]
            f1, accuracy = [np.asarray([row[k] for row in group]) for k in ("macro_f1_percent", "accuracy_percent")]
            summary.append({"method": method, "condition": name, "macro_f1_mean_percent": float(f1.mean()),
                            "macro_f1_sd_percent": float(f1.std(ddof=1)), "accuracy_mean_percent": float(accuracy.mean()),
                            "accuracy_sd_percent": float(accuracy.std(ddof=1))})
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    exp.write_csv(args.output / "reports/PER_SEED_RESULTS.csv", rows)
    exp.write_csv(args.output / "reports/PER_CLASS_PER_SEED.csv", per_class)
    exp.write_csv(args.output / "reports/SUMMARY.csv", summary)
    exp.write_json(args.output / "reports/SUMMARY.json", {"protocol": config["protocol"], "summary": summary, "seeds": config["seeds"]})
    exp.write_json(args.output / "reports/WEIGHTS_AND_RUN_FILES_SHA256.json", protected)
    exp.write_json(args.output / "reports/VERIFICATION.json", {"status": "passed", "new_models": 9,
                   "fixed_reference_models": 6, "target_evaluation_sets": len(rows), "new_prediction_sets_exactly_replayed": 99,
                   "historical_reference_predictions_exactly_reproduced": len(checked), "reference_replays": checked,
                   "source_selected_prediction_sets_replayed": 54, "all_target_metrics_independently_recomputed": True,
                   "original_target_cohort_and_labels_unchanged": True, "target_dev_records": 88, "target_dev_cycles": 2474,
                   "feature_dimension": 45, "feature_reconstruction_errors": feature_errors,
                   "all_model_and_run_files_sha_unchanged": len(protected), "reference_model_parameters_and_buffers_unchanged": True,
                   "target_training_or_selection": False, "final_holdout_evaluated": False,
                   "elapsed_seconds": time.perf_counter() - started})
    print("EVALUATION_COMPLETE: 165 Dev sets, all source-only selections and reference replays verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--reference-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/lp4_task_difficulty_v1.json")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    for name in ("source-strong-root", "target-snr-root", "target-strong-root", "minus5-root", "reference-aprime", "reference-results", "reference-minus5-results"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_code / "scripts"))
    import hust_arch_suite_experiment as exp
    config = read(args.config)
    assert torch.cuda.is_available(), "实验在原服务器CUDA环境执行"
    exp.fix_randomness(42)
    if args.stage == "train":
        assert args.seed is not None and args.source_strong_root is not None
        train(args, exp, config)
    else:
        evaluate(args, exp, config)
