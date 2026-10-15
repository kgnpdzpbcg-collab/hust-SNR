"""只读审计A0/B0：选模、种子差异、小波描述符、复谐波恢复与固定权重干预。

输入：原服务器实验目录及独立输出目录。所有模型从原best.pt载入，不训练、
不更新BN、不改变正式选模；仅使用原Source-Train/Val和88条实测Dev。
输出：诊断CSV/JSON和原文件SHA。干预结果仅用于定位当前权重的敏感性，
不能视作重训消融成绩或独立Test结论，也不用于选择新超参数。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")
import numpy as np
import torch
from torch.nn import functional as F


def read(path):
    """读取冻结记录，不重写原文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """分块校验权重和记录字节，审计结束时再次核验。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def tensor_summary(values):
    """保留均值及分位数；明确是周期描述统计，不是独立样本置信区间。"""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    assert np.isfinite(x).all()
    return {"mean": float(x.mean()), "p05": float(np.quantile(x, .05)),
            "median": float(np.median(x)), "p95": float(np.quantile(x, .95))}


def project(x, maximum):
    """保留复谱0..maximum阶，不改变其幅值或相位。"""
    spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")
    spectrum[..., maximum + 1:] = 0
    return torch.fft.irfft(spectrum, n=4096, dim=-1, norm="ortho")


def classify_restored(model, restored):
    """复用同一B0分类器，只执行一次原MD；不修改任何模块或参数。"""
    signal = model.normalize(restored)
    features = model.backbone.pool(model.backbone.layers(model.backbone.stem(signal))).flatten(1)
    return model.backbone.classifier(features)


@torch.no_grad()
def infer_mode(model, x, mode="original", shift=0):
    """固定权重的输入/频带干预；返回预测及平均CE所需的完整六类logits。"""
    parts = []
    for pos in range(0, len(x), 128):
        raw = torch.from_numpy(x[pos:pos + 128]).cuda()
        if shift:
            raw = torch.roll(raw, shifts=shift, dims=-1)
        if mode == "original":
            scores = model(raw)
        elif mode == "a0_input_lp4":
            scores = model(project(raw, 4))
        elif mode in ("b0_observed_k8", "b0_observed_k4"):
            scores = classify_restored(model, project(raw, 8 if mode.endswith("8") else 4))
        elif mode == "b0_learned_k4":
            scores = classify_restored(model, project(model.restorer(raw), 4))
        else:
            raise ValueError(mode)
        parts.append(scores.cpu())
    logits = torch.cat(parts)
    return logits.argmax(1).numpy(), logits


def balanced_indices(y, each=12):
    """固定取每类相同数量周期，避免结构探针仅取到文件开头的健康段。"""
    return np.concatenate([np.flatnonzero(y == label)[:each] for label in range(6)])


def history_audit(reference):
    """逐轮定位满分、并列和最终保存轮次；训练损失与验证指标分别记录。"""
    rows = []
    for method in ("a0", "b0"):
        for seed in (42, 43, 44):
            folder = reference / f"results/models/{method}/seed{seed}"
            history, run = read(folder / "history.json"), read(folder / "run.json")
            selected = run["selected"]
            full = [r["epoch"] for r in history if r["clean_f1"] == 1 and all(v == 1 for v in r["noisy_f1"].values())]
            ties = [r["epoch"] for r in history if (r["noise_mean_f1"], r["clean_f1"]) ==
                    (selected["noise_mean_f1"], selected["clean_f1"])]
            row = {"method": method, "seed": seed, "selected_epoch": selected["epoch"],
                   "first_all_six_validation_full_epoch": full[0] if full else None,
                   "all_six_validation_full_epoch_count": len(full), "equal_selected_score_epochs": ties,
                   "selected": selected, "first_five_epochs": history[:5], "final_epoch": history[-1]}
            if method == "b0":
                row["pretrain_recovery_losses"] = run["pretrain_recovery_losses"]
                row["selected_weighted_reconstruction_to_ce_loss_ratio"] = .2 * selected["source_recovery_loss_mean"] / selected["classification_ce_mean"]
            rows.append(row)
    return rows


@torch.no_grad()
def wavelet_probe(model, x):
    """检查第一层有符号GAP能否表示细节能量，以及实际注意力增益范围。"""
    from hust_a0_wavelet_attention import haar_dwt
    raw = torch.from_numpy(x).cuda()
    signal = model.normalize(raw)
    low, high = haar_dwt(signal)
    features = torch.cat((low, high), dim=1)
    weights = 1 + model.layers[0].analysis.fam(features.mean(-1))
    rms = high.square().mean(-1).sqrt().clamp_min(1e-12)
    # 人工反例：两组细节能量相差100倍，但所有通道有符号GAP都精确为零。
    example = raw.new_tensor([1., -1., -1., 1.]).repeat(1, 5, 128)
    first = torch.cat(haar_dwt(example), dim=1).mean(-1)
    second = torch.cat(haar_dwt(example * 10), dim=1).mean(-1)
    assert torch.equal(first, second) and torch.count_nonzero(first) == 0
    attention_difference = (model.layers[0].analysis.fam(first) - model.layers[0].analysis.fam(second)).abs().max()
    return {"detail_abs_signed_gap_over_rms": tensor_summary((high.mean(-1).abs() / rms).cpu().numpy()),
            "attention_gain": tensor_summary(weights.cpu().numpy()),
            "synthetic_detail_energy_ratio": 100,
            "synthetic_same_gap_attention_max_abs_difference": float(attention_difference.cpu())}


@torch.no_grad()
def recovery_probe(model, noisy, clean, y):
    """用同周期干净参考检查恢复误差；同时核对0.2A弱电流分支的变化。

    干净目标只在本诊断计算误差，绝不进入正式模型forward或权重选择。
    误差按干净五相基波参考归一化，避免工况幅值直接支配统计。
    """
    errors, branches, gains = {}, {}, []
    gain_hook = model.restorer.gain.register_forward_hook(lambda module, inputs, output: gains.append(torch.sigmoid(output).cpu().numpy()))
    collected = {key: [] for key in ("observed_k8", "learned_k8", "observed_k4")}
    branch_all = {key: [] for key in ("clean_k8", "observed_k8", "learned_k8", "observed_k4")}
    try:
        for pos in range(0, len(noisy), 128):
            nx, cx = torch.from_numpy(noisy[pos:pos + 128]).cuda(), torch.from_numpy(clean[pos:pos + 128]).cuda()
            ref = torch.fft.rfft(cx, dim=-1, norm="ortho")
            obs = torch.fft.rfft(nx, dim=-1, norm="ortho")
            restored, estimated = model.restorer(nx, return_spectrum=True)
            scale = model.restorer.reference_scale(ref)[:, None, None]
            before = (obs[..., :9] - ref[..., :9]) / scale
            after = (estimated - ref[..., :9]) / scale
            lowpass4 = obs[..., :9].clone()
            lowpass4[..., 5:] = 0
            error4 = (lowpass4 - ref[..., :9]) / scale
            for key, value in (("observed_k8", before), ("learned_k8", after), ("observed_k4", error4)):
                collected[key].append(value.abs().square().mean((1, 2)).cpu().numpy())
            for key, value in (("clean_k8", project(cx, 8)), ("observed_k8", project(nx, 8)),
                               ("learned_k8", restored), ("observed_k4", project(nx, 4))):
                centered = value - (value.amax(-1, keepdim=True) + value.amin(-1, keepdim=True)) / 2
                branch_all[key].append((centered.abs().mean(-1) < .2).cpu().numpy())
    finally:
        gain_hook.remove()
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    weak = {key: np.concatenate(value) for key, value in branch_all.items()}
    for key, value in arrays.items():
        errors[key] = {"all_cycles": tensor_summary(value), "by_class_mean": [float(value[y == label].mean()) for label in range(6)]}
    for key, value in weak.items():
        branches[key] = {"weak_phase_fraction": value.mean(0).tolist(),
                         "different_from_clean_fraction": float(np.mean(value != weak["clean_k8"])),
                         "open_A_false_active_fraction": float(np.mean(~value[y == 3, 0])),
                         "open_C_false_active_fraction": float(np.mean(~value[y == 5, 2]))}
    return {"normalized_complex_mse": errors, "md_weak_phase_branch": branches,
            "gain_distribution": tensor_summary(np.concatenate(gains))}


def gradient_probe(exp, model, clean, noisy, y):
    """只求梯度不更新参数，比较CE和0.2×恢复项对恢复器的实际梯度量级。"""
    model.requires_grad_(True).eval()
    idx = balanced_indices(y, each=5)
    cx, nx = torch.from_numpy(clean[idx]).cuda(), torch.from_numpy(noisy[idx]).cuda()
    labels = torch.from_numpy(y[idx]).cuda()
    logits, spectrum = model(torch.cat((cx, nx)), return_recovery=True)
    ce = .5 * (F.cross_entropy(logits[:len(idx)], labels) + F.cross_entropy(logits[len(idx):], labels))
    rec = .2 * exp.harmonic_recovery_loss(spectrum[len(idx):], cx)
    params = list(model.restorer.parameters())
    first = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(rec, params, allow_unused=True)
    flatten = lambda gradients: torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) for g, p in zip(gradients, params)])
    a, b = flatten(first), flatten(second)
    model.requires_grad_(False)
    return {"balanced_cycles": len(idx), "ce": float(ce.detach()), "weighted_rec": float(rec.detach()),
            "ce_restorer_gradient_norm": float(a.norm()), "weighted_rec_gradient_norm": float(b.norm()),
            "rec_to_ce_gradient_norm_ratio": float(b.norm() / a.norm().clamp_min(1e-20)),
            "gradient_cosine": float(F.cosine_similarity(a[None], b[None]).item())}


def main(args):
    """全部诊断写入新目录，原模型、结果与代码SHA始终保持不变。"""
    started = time.perf_counter()
    assert not args.output.exists()
    reference = args.reference_root
    base = reference.parents[2]
    sys.path.insert(0, str(reference / "code/scripts"))
    import hust_prototypes_experiment as exp
    from hust_arch_suite_models import HUSTArchitectureCandidate
    from hust_b0_harmonic_recovery import HarmonicRecovery, complex_to_channels, channels_to_complex
    from prepare import OUT
    config = read(reference / "DEPLOYMENT_MANIFEST.json")
    assert read(reference / "results/reports/VERIFICATION.json")["status"] == "passed"
    protected = {str(p): sha(p) for p in (reference / "results").rglob("*") if p.is_file()}
    protected.update({str(reference / "code" / name): value for name, value in config["code_sha256"].items()})
    (tr, tm, ty, train_clean), (vr, vm, vy, val_clean, val_noise) = exp.source_views(Path(config["source_strong_root"]))
    train_ids, val_ids = set(tr), set(vr)
    assert train_ids.isdisjoint(val_ids)
    hashes = {hashlib.sha256(x.tobytes()).hexdigest() for x in train_clean}
    shared_hash_count = sum(hashlib.sha256(x.tobytes()).hexdigest() in hashes for x in val_clean)
    data = {"source_val_clean": (val_clean, val_clean, vy),
            "source_val_0": (val_noise[0], val_clean, vy), "source_val_10": (val_noise[10], val_clean, vy)}
    frozen = read(OUT / "target_dev_clean_windows.json")
    truth = np.asarray([r["label"] for r in frozen], dtype=np.int64)
    clean_raw, clean_meta = exp.data.target_dev_from_mat(None, None)
    assert clean_meta == frozen
    target_clean = exp.data.raw_cycles(clean_raw, clean_meta)
    data["target_clean"] = (target_clean, target_clean, truth)
    for snr in exp.CONDITIONS:
        raw, meta = exp._target_cycles(snr, ordinary_root=Path(config["target_strong_root"]),
                        extreme_root=Path(config["target_extreme_root"]), export_root=None)
        assert meta == frozen and len(raw) == 88
        data[f"target_{snr}"] = (exp.data.raw_cycles(raw, meta), target_clean, truth)
    args.output.mkdir(parents=True)
    exp.write_json(args.output / "HISTORY_AUDIT.json", history_audit(reference))
    metrics_rows, shifts, probes, gradients, noise_rows, confidence = [], [], [], [], [], []
    # 输入噪声代理的分布单独测量，不把其误称为真实SNR估计。
    with torch.no_grad():
        for condition, (x, clean, labels) in data.items():
            parts = []
            for pos in range(0, len(x), 128):
                spectrum = torch.fft.rfft(torch.from_numpy(x[pos:pos + 128]).cuda(), norm="ortho")
                proxy = HarmonicRecovery.noise_features(spectrum, HarmonicRecovery.reference_scale(spectrum))
                parts.append(proxy.cpu().numpy())
            noise_rows.append({"condition": condition, **tensor_summary(np.concatenate(parts))})
    for method in ("a0", "b0", "resnet_lp4"):
        for seed in exp.SEEDS:
            exp.fix_randomness(seed)
            root = reference if method != "resnet_lp4" else base / "experiments/architecture_suite/5c75bea"
            weight = root / f"results/models/{method}/seed{seed}/best.pt"
            protected[str(weight)] = sha(weight)
            saved = torch.load(weight, map_location="cpu", weights_only=True)
            model = exp.build_model(method) if method != "resnet_lp4" else HUSTArchitectureCandidate(method)
            model = model.cuda().eval().requires_grad_(False)
            model.load_state_dict(saved["state"], strict=True)
            for condition, (x, clean, labels) in data.items():
                modes = ["original"]
                if method == "a0":
                    modes += ["a0_input_lp4"]
                elif method == "b0":
                    modes += ["b0_observed_k8", "b0_observed_k4", "b0_learned_k4"]
                original = None
                for mode in modes:
                    predicted, logits = infer_mode(model, x, mode)
                    score = exp.metrics(labels, predicted)
                    metrics_rows.append({"method": method, "seed": seed, "condition": condition, "mode": mode,
                              "macro_f1_percent": 100 * score["macro_f1"], "accuracy_percent": 100 * score["accuracy"],
                              "per_class_f1_percent": [100 * v for v in score["per_class_f1"]],
                              "per_class_recall_percent": [100 * v for v in score["per_class_recall"]],
                              "confusion_matrix": score["confusion_matrix"]})
                    if mode == "original":
                        original = predicted
                        confidence.append({"method": method, "seed": seed, "condition": condition,
                                "cross_entropy": float(F.cross_entropy(logits, torch.from_numpy(labels))),
                                "mean_max_probability": float(logits.softmax(1).max(1).values.mean())})
                        if method != "resnet_lp4" and condition.startswith("target_") and condition != "target_clean":
                            snr = condition.split("_", 1)[1]
                            old = read(reference / f"results/evaluations/{method}/seed{seed}/target_dev_snr{snr}.json")
                            assert np.array_equal(predicted, old["predicted"])
                # 改周期起点不改变五相相对关系和类别；记录固定分类器的敏感性。
                if condition in ("source_val_clean", "target_clean", "target_-10"):
                    for shift in (512, 1024, 2048):
                        predicted, _ = infer_mode(model, x, shift=shift)
                        score = exp.metrics(labels, predicted)
                        shifts.append({"method": method, "seed": seed, "condition": condition,
                                       "shift_samples": shift, "prediction_changed_percent": 100 * float(np.mean(predicted != original)),
                                       "macro_f1_percent": 100 * score["macro_f1"]})
                if method == "a0":
                    idx = balanced_indices(labels)
                    probes.append({"method": method, "seed": seed, "condition": condition,
                                   "probe": wavelet_probe(model, x[idx])})
                elif method == "b0":
                    probes.append({"method": method, "seed": seed, "condition": condition,
                                   "probe": recovery_probe(model, x, clean, labels)})
            if method == "b0":
                noisy = exp.data.fresh_noisy_source(tr, tm, seed, 1, noise_levels=exp.TRAIN_NOISE)
                for condition, cx, nx, y in (("train_first_epoch_noise", train_clean, noisy, ty),
                                             ("source_val_0", val_clean, val_noise[0], vy)):
                    gradients.append({"seed": seed, "condition": condition,
                                      **gradient_probe(exp, model, cx, nx, y)})
                # 只计调用，不改模块：B0前向必须恰好调用一次MD归一化。
                calls = []
                hooks = [module.register_forward_hook(lambda *a: calls.append(1)) for module in model.modules()
                         if type(module).__name__ == "MDNormalization"]
                with torch.no_grad():
                    model(torch.from_numpy(val_clean[:2]).cuda())
                for hook in hooks:
                    hook.remove()
                assert len(calls) == 1
            assert all(torch.equal(value.detach().cpu(), saved["state"][name]) for name, value in model.state_dict().items())
            print(f"AUDIT_COMPLETE {method} seed{seed}", flush=True)
            del model
    assert all(sha(path) == expected for path, expected in protected.items())
    flat_metrics = [{k: v for k, v in row.items() if k not in ("per_class_f1_percent", "per_class_recall_percent", "confusion_matrix")} for row in metrics_rows]
    for filename, rows in (("FIXED_WEIGHT_INTERVENTIONS.csv", flat_metrics), ("CYCLIC_SHIFT_SENSITIVITY.csv", shifts),
                           ("NOISE_PROXY_DISTRIBUTION.csv", noise_rows), ("CONFIDENCE_AND_CE.csv", confidence),
                           ("RESTORER_GRADIENTS.csv", gradients)):
        exp.write_csv(args.output / filename, rows)
    exp.write_json(args.output / "FIXED_WEIGHT_INTERVENTIONS.json", metrics_rows)
    exp.write_json(args.output / "STRUCTURAL_PROBES.json", probes)
    exp.write_json(args.output / "PROTECTED_FILES_SHA256.json", protected)
    exp.write_json(args.output / "VERIFICATION.json", {"status": "passed", "source_train_records": len(train_ids),
                    "source_val_records": len(val_ids), "source_train_val_record_overlap": len(train_ids & val_ids),
                    "source_val_raw_cycles_exactly_duplicate_training": shared_hash_count,
                    "source_val_raw_cycles_checked": len(val_clean), "official_target_prediction_sets_replayed": 24,
                    "fixed_weights_checked": 9, "models_trained": 0, "optimizer_updates": 0,
                    "all_weight_parameters_and_buffers_unchanged": True,
                    "protected_files_sha_unchanged": len(protected), "b0_md_normalization_calls_per_forward": 1,
                    "target_dev_records": 88, "target_dev_cycles": 2474, "final_holdout_accessed": False,
                    "interventions_used_for_model_or_parameter_selection": False,
                    "later_epoch_weights_available_in_original_run": False,
                    "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha(Path(__file__))})
    print("A0_B0_FIXED_WEIGHT_AUDIT_ALL_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
