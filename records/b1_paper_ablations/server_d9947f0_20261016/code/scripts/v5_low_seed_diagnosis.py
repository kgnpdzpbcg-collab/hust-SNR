"""V5低分种子诊断：固定权重的门控和源域BatchNorm反事实，不优化参数。

正常成绩来自原最佳权重。所有干预结果另存独立诊断目录，不能用于重新
选模或替换正式结果。BN重算只读取仿真训练电流，不使用验证/目标域统计。
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn


class ConstantFusion(nn.Module):
    """只改变第四阶增益；保持V5的FFT算子、五相共享掩码和复数相位。"""
    def __init__(self, alpha):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("常数α必须在0到1之间")
        self.alpha = float(alpha)

    def forward(self, raw):
        """输入/输出为[B,5,4096]原始周期，CNN和归一化未修改。"""
        if raw.shape[1:] != (5, 4096):
            raise ValueError("需要五相4096点周期")
        x = torch.fft.rfft(raw, dim=-1, norm="ortho")
        mask = raw.new_zeros((len(raw), x.shape[-1]))
        mask[:, :4] = 1.
        mask[:, 4] = self.alpha
        return torch.fft.irfft(x*mask.unsqueeze(1), n=4096, dim=-1, norm="ortho")


def parameter_sha(model):
    """仅哈希可学习参数；BN运行均值/方差属于缓冲区，单独记录变化。"""
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


@torch.no_grad()
def calibrate_source_bn(model, source, batch_size=128):
    """逐层用全部仿真训练输入重算精确统计，权重/门控/BN仿射参数不变。

    每层按eval模式累积其输入的逐通道sum/sumsq/count；先校准第一层，
    再在上游已校准条件下校准下一层。统计按所有样本/时间点加权，避免
    最后小批次被等权放大；方差用n/(n-1)修正，与BN运行无偏方差一致。
    source是Clean和该模型最佳轮次训练噪声拼接，调用者不能传目标数据。
    """
    cloned = copy.deepcopy(model).eval()
    before = parameter_sha(cloned)
    changes = []
    for name, bn in cloned.named_modules():
        if not isinstance(bn, nn.BatchNorm1d):
            continue
        sums = torch.zeros(bn.num_features, device="cuda", dtype=torch.float64)
        squares = sums.clone()
        count = 0

        def collect(_module, inputs):
            """读取当前BN之前的激活，绝不更新目标域或其他运行状态。"""
            nonlocal count
            value = inputs[0].double()
            sums.add_(value.sum(dim=(0, 2)))
            squares.add_(value.square().sum(dim=(0, 2)))
            count += value.shape[0]*value.shape[2]

        hook = bn.register_forward_pre_hook(collect)
        for start in range(0, len(source), batch_size):
            cloned(torch.from_numpy(source[start:start+batch_size]).cuda())
        hook.remove()
        mean = sums/count
        variance = (squares/count-mean.square()).clamp_min(0)*count/(count-1)
        old_mean, old_var = bn.running_mean.clone(), bn.running_var.clone()
        bn.running_mean.copy_(mean.to(bn.running_mean.dtype))
        bn.running_var.copy_(variance.to(bn.running_var.dtype))
        changes.append({"layer": name, "count_per_channel": count,
                        "mean_change_rms": float((old_mean-bn.running_mean).square().mean().sqrt().cpu()),
                        "old_var_mean": float(old_var.mean().cpu()), "new_var_mean": float(bn.running_var.mean().cpu())})
    assert parameter_sha(cloned) == before and len(changes) == 3
    return cloned, changes


@torch.no_grad()
def signal_diagnostics(model, inputs, truth, prediction, batch_size=128):
    """检查门控对MD阈值/归一化输入的影响，仅输出统计，不导出电流。

    同一raw分别使用原α和固定0.5，记录低电流分支翻转数量及归一化后
    相对L2变化。分开统计错误和正确周期，标签只用于事后分组。
    """
    relative, flips, alphas, descriptors = [], [], [], []
    fixed = ConstantFusion(.5).cuda()
    for start in range(0, len(inputs), batch_size):
        raw = torch.from_numpy(inputs[start:start+batch_size]).cuda()
        shaped, alpha, z = model.shaping(raw, return_descriptors=True)
        other = fixed(raw)
        centered = lambda x: x-(x.amax(-1, keepdim=True)+x.amin(-1, keepdim=True))*.5
        flags, other_flags = (centered(x).abs().mean(-1)<.2 for x in (shaped, other))
        a, b = model.normalize(shaped), model.normalize(other)
        rel = (a-b).flatten(1).norm(dim=1)/a.flatten(1).norm(dim=1).clamp_min(1e-12)
        relative.append(rel.cpu().numpy())
        flips.append((flags != other_flags).sum(dim=1).cpu().numpy())
        alphas.append(alpha.cpu().numpy())
        descriptors.append(z.cpu().numpy())
    relative, flips, alpha, z = map(np.concatenate, (relative, flips, alphas, descriptors))
    def group(indices):
        if not indices.any():
            return {"cycles": 0}
        return {"cycles": int(indices.sum()), "alpha_mean": float(alpha[indices].mean()),
                "alpha_min": float(alpha[indices].min()), "alpha_max": float(alpha[indices].max()),
                "relative_input_l2_mean": float(relative[indices].mean()),
                "relative_input_l2_p95": float(np.quantile(relative[indices], .95)),
                "md_branch_flip_cycles": int((flips[indices]>0).sum()),
                "md_branch_flip_phases": int(flips[indices].sum()),
                "descriptor_mean": z[indices].mean(0).astype(float).tolist()}
    return {"all": group(np.ones(len(inputs), dtype=bool)), "wrong": group(prediction != truth),
            "correct": group(prediction == truth), "by_true_class": [group(truth==k) for k in range(6)]}


def main():
    """真实权重重放与预先声明干预，原文件受SHA保护，不改正式模型或成绩。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-run", type=Path, required=True)
    parser.add_argument("--four-run", type=Path, required=True)
    parser.add_argument("--strong-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-seed", type=int, default=2025)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("拒绝覆盖诊断结果")
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.extra_run / "code/scripts"))
    import v2_experiment as data
    import five_models_extra_seeds as experiment
    from v1_noise_consistency import fix_randomness, metrics
    from v5_experiment import predict_with_gate
    from evaluate_v4_strong_snr import read, write, csv_write, frozen_ids
    from prepare import OUT, ROOT as DATA_ROOT
    seeds = experiment.ALL_SEEDS
    if args.focus_seed not in seeds:
        raise ValueError("重点种子必须来自当前六个种子")
    protected = read(args.extra_run / "ORIGINAL_FILES_SHA256.json")
    protected.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for run in (args.extra_run, args.four_run)
                      for p in (run / "results").rglob("*") if p.is_file()})
    write(args.output / "PROTECTED_FILES_SHA256.json", protected)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items())
    tr, tm, ty = data.source_data("train")
    vr, vm, vy = data.source_data("val")
    source_clean = data.raw_cycles(tr, tm)
    source_val = {"source_val_clean": data.raw_cycles(vr, vm)}
    source_val.update({f"source_val_snr{k}": x for k, x in data.fixed_noisy_validation(vr, vm,
         noise_levels=(0, 5, 10, 20, 30), materialized_root=args.strong_data_root/"generated/source").items()})
    metadata = read(OUT / "target_dev_clean_windows.json")
    frozen_ids(read(DATA_ROOT / "configs/splits_gate0_v1.json"), metadata)
    labels = read(args.labels)["experimental"]
    variants, summaries, calibration, weight_hashes, selections = {}, [], {}, {}, {}
    for seed in seeds:
        fix_randomness(seed)
        root = args.extra_run if seed in experiment.NEW_SEEDS else args.four_run
        path = root / f"results/models/nahsf/seed{seed}/best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if seed in experiment.NEW_SEEDS:
            experiment.common.check_checkpoint(checkpoint, "nahsf", seed, protocol=experiment.config(), version=experiment.VERSION)
        else:
            experiment.common.check_checkpoint(checkpoint, "nahsf", seed)
        selections[seed] = checkpoint["selected"]
        model = experiment.make_model("nahsf").cuda().eval()
        model.load_state_dict(checkpoint["state"], strict=True)
        weight_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        epoch = checkpoint["selected"]["epoch"]
        noisy = data.fresh_noisy_source(tr, tm, seed, epoch)
        source = np.concatenate((source_clean, noisy))
        _, alpha, _ = predict_with_gate(model, source, torch.device("cuda"))
        calibrated, changes = calibrate_source_bn(model, source)
        calibration[str(seed)] = {"source_records": len(tr), "source_cycles_clean_plus_noisy": len(source),
                                 "noise_seed": seed, "noise_epoch": epoch, "alpha_source_mean": float(alpha.mean()),
                                 "layer_changes": changes, "learnable_parameters_unchanged": True}
        variants[seed, "native"] = model
        forced = copy.deepcopy(model)
        forced.shaping = ConstantFusion(.5).cuda()
        variants[seed, "alpha05"] = forced
        variants[seed, "bn_source_all"] = calibrated
        if seed == args.focus_seed:
            constant = copy.deepcopy(model)
            constant.shaping = ConstantFusion(float(alpha.mean())).cuda()
            variants[seed, "alpha_source_mean"] = constant
            original_state, calibrated_state = model.state_dict(), calibrated.state_dict()
            for layer in changes:
                single = copy.deepcopy(model)
                state = dict(original_state)
                name = layer["layer"]
                for suffix in ("running_mean", "running_var"):
                    state[name+"."+suffix] = calibrated_state[name+"."+suffix]
                single.load_state_dict(state, strict=True)
                variants[seed, "bn_source_"+name.split(".")[-1]] = single
        # 训练域eval与batch-statistics对照，仅在仿真训练数据上使用train模式。
        native_eval = data.inference(model, source, torch.device("cuda"))
        batch_model = copy.deepcopy(model).train()
        before_parameters = parameter_sha(batch_model)
        order = np.random.default_rng(seed+901+epoch).permutation(len(ty))
        predicted = np.empty((2, len(ty)), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, len(ty), 32):
                indices = order[start:start+32]
                raw = np.concatenate((source_clean[indices], noisy[indices]))
                result = batch_model(torch.from_numpy(raw).cuda()).argmax(1).cpu().numpy()
                predicted[0, indices], predicted[1, indices] = np.split(result, 2)
        assert before_parameters == parameter_sha(batch_model)
        calibration[str(seed)]["source_train_native_eval"] = metrics(np.tile(ty, 2), native_eval)
        calibration[str(seed)]["source_train_batch_stats"] = metrics(np.tile(ty, 2), predicted.reshape(-1))
        del batch_model
    # 先核验仿真验证正常重放，后执行四个实测条件；不能按目标成绩选变体。
    for condition, inputs in source_val.items():
        for (seed, variant), model in variants.items():
            prediction = data.inference(model, inputs, torch.device("cuda"))
            metric = metrics(vy, prediction)
            if variant == "native":
                expected = selections[seed]["clean_f1"] if condition == "source_val_clean" else selections[seed]["noisy_f1"][condition.removeprefix("source_val_snr")]
                assert abs(metric["macro_f1"]-expected)<1e-12
            summaries.append({"seed": seed, "variant": variant, "condition": condition,
                              "macro_f1_percent": 100*metric["macro_f1"], "accuracy_percent": 100*metric["accuracy"]})
    detailed, records, signal = [], [], []
    for level in (None, 10, 5, 0):
        condition = "clean" if level is None else f"snr{level}"
        snr_root = args.strong_data_root/"generated/target" if level in (0, 5) else args.extra_run/"target_snr"
        raw, meta = data.target_dev_from_mat(level, snr_root)
        assert meta == metadata
        inputs = data.raw_cycles(raw, metadata)
        truth = np.array([r["label"] for r in meta])
        for (seed, variant), model in variants.items():
            prediction = data.inference(model, inputs, torch.device("cuda"))
            metric = metrics(truth, prediction)
            base = {"seed": seed, "variant": variant, "condition": condition}
            if variant == "native":
                root = args.extra_run if seed in experiment.NEW_SEEDS else args.four_run
                original = read(root/f"results/evaluations/nahsf/seed{seed}/target_dev_{condition}.json")
                assert np.array_equal(original["predicted"], prediction)
                assert metric == original["metrics"]
                signal.append({**base, "statistics": signal_diagnostics(model, inputs, truth, prediction)})
            summaries.append({**base, "macro_f1_percent": 100*metric["macro_f1"], "accuracy_percent": 100*metric["accuracy"]})
            detailed.append({**base, "metrics": metric, "truth": truth.tolist(), "predicted": prediction.tolist(),
                             "diagnostic_only": True, "official_results_changed": False})
            for rid in sorted({r["record_id"] for r in meta}):
                indices = np.array([r["record_id"] == rid for r in meta])
                fault = indices & (truth>0)
                counts = np.bincount(prediction[fault], minlength=6)
                records.append({**base, "record_id": rid, "filename": rid+".mat", "speed": labels[rid]["speed"],
                                "torque": labels[rid]["torque"], "fault_label": labels[rid]["fault_label"],
                                "fault_name": labels[rid]["fault_name"], "cycles": int(indices.sum()),
                                "errors": int((prediction[indices] != truth[indices]).sum()), "fault_cycles": int(fault.sum()),
                                "fault_errors": int((prediction[fault] != truth[fault]).sum()),
                                **{f"fault_predicted_{k}": int(counts[k]) for k in range(6)}})
        print(f"DIAGNOSTIC_COMPLETED {condition}: {len(variants)} frozen-weight variants", flush=True)
    csv_write(args.output/"METRICS.csv", summaries)
    csv_write(args.output/"PER_RECORD_ERRORS.csv", records)
    write(args.output/"TARGET_DIAGNOSTICS.json", detailed)
    write(args.output/"SIGNAL_DIAGNOSTICS.json", signal)
    write(args.output/"BN_SOURCE_CALIBRATION.json", calibration)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items())
    write(args.output/"STATUS.json", {"status": "passed", "focus_seed": args.focus_seed, "seeds": list(seeds),
          "variants": [{"seed": s, "variant": v} for s, v in variants], "native_prediction_replays": 24,
          "source_calibration_records": 34, "source_validation_records": 15, "target_dev_records": 88,
          "target_conditions": ["clean", 10, 5, 0], "weights_sha256": weight_hashes,
          "bn_calibration_used_target_or_val": False, "optimizer_updates": 0,
          "old_results_and_weights_unchanged": True, "final_holdout_evaluated": False,
          "diagnostic_not_official_results": True})


if __name__ == "__main__":
    main()
