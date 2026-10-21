"""Jung (TPEL 2025) / Hang (TIE 2023) 高保真复现 V6 运行入口。

与 V5 的区别只在"输入"和"加噪"两层的保真修正，网络结构与原有训练/选模/
评价协议保持一致。两个模型都使用单电周期样本，因此源域仍为 962/221，
实测 Dev 仍为 2474 个窗口，与其余八个模型共用一个评价队列。

阶段：
    preflight  源域输入审计（图像退化、THD、注噪覆盖率）与前向自检
    train      单模型单种子源域训练
    evaluate   七档固定 Dev 评价
    verify     重放核验（预测、源验证、选模记录）
    summarize  汇总并与其他模型合表

不读取 347 条 Final-Holdout；目标数据不参与训练或选模。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LOCAL = Path(__file__).resolve().parent
CONFIG = LOCAL.parent / "configs/jung_hang_v6_v1.json"
if str(LOCAL) not in sys.path:
    sys.path.insert(0, str(LOCAL))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    """只写本轮输出；固定 LF 便于两端逐字节核对。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf8", newline="\n",
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(args):
    """载入冻结配置、源划分与参考数据根目录。"""
    config = read(args.config)
    if not config["models"] or not set(config["models"]).issubset({"jung", "hang"}):
        raise ValueError("models must be a nonempty Jung/Hang subset")
    if config["seeds"] != [42, 44]:
        raise ValueError("V6 uses the paper seeds 42/44")

    import compare_jung_hang_v6 as paper
    import source_cycle_benchmark as ref
    import source_cycle_split as split

    provenance = split.source_provenance(args.source_root)
    if provenance["manifest_sha256"] != config["source_split_manifest_sha256"]:
        raise ValueError("source split manifest changed")

    manifest_path = args.reference_root / "DEPLOYMENT_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return config, ref, paper, split, provenance, read(manifest_path)


def source_data(args, method):
    """构造源域输入。

    返回：
        config / modules / provenance / raw / metadata / labels /
        clean_inputs / val_labels / {clean, snrX: inputs}
    """
    import numpy as np

    config, ref, paper, split, provenance, _ = setup(args)
    (raw, meta, labels, _), (_, val_meta, val_labels, _, _) = split.source_views(args.source_root)
    if (len(labels), len(val_labels)) != (962, 221):
        raise ValueError("source sample count changed")

    clean, train_positions = paper.build_inputs(method, meta, raw, config)
    val_clean, val_positions = paper.build_inputs(method, val_meta, raw, config)
    # 单周期窗口不跨分组也不丢锚点，位置必须与标签逐行对齐。
    np.testing.assert_array_equal(train_positions, np.arange(len(meta)))
    np.testing.assert_array_equal(val_positions, np.arange(len(val_meta)))

    val = {"clean": val_clean}
    for snr in config["val_noise_db"]:
        modified = noise_records(raw, 0, 0, method=method, meta=val_meta,
                                 config=config, snr=snr)
        val_input, positions = paper.build_inputs(method, val_meta, modified, config)
        np.testing.assert_array_equal(positions, val_positions)
        val[f"snr{snr}"] = val_input
    return config, ref, paper, provenance, raw, meta, labels, clean, val_labels, val


def noise_records(raw, seed, epoch, *, method, meta, config, snr=None):
    """生成一轮训练噪声（或固定验证噪声）。

    Jung：按原文式 (5) 注入到目标 THD+N，并对被阻断相区段屏蔽噪声。
    Hang：按原文式 (17) 逐相注入 15/20 dB 白噪声。

    验证噪声统一使用 SNR 口径，与七档实测条件同一噪声族，使选模目标与
    部署噪声一致；训练噪声则分别遵守各自原论文的规定。
    """
    import numpy as np

    from jung_hang_models_v6 import (
        add_awgn, jung_thdn_augment, record_reference,
    )

    blocked = {}
    for row in meta:
        blocked.setdefault(row["record_id"], []).append(row)

    output = {}
    for index, rid in enumerate(sorted(raw)):
        time, currents, omega = raw[rid]
        if snr is None:
            rng = np.random.default_rng(seed + 15000 + epoch + 997 * (index + 1))
            levels = ([config["jung_thdn_percent"]] if method == "jung"
                      else list(config["hang_train_snr_db"]))
            level = levels[(epoch + index + seed) % len(levels)]
        else:
            digest = hashlib.sha256(f"HUST-V6:{method}:val:{snr}:{rid}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            level = snr

        # Jung只有训练视图使用THD+N；两模型的固定验证视图均按snr_db加AWGN。
        # 否则0 dB会被误当成0% THD+N，得到未加噪数据并改变源域选模。
        if method == "jung" and snr is None:
            dc, _, _ = record_reference(time, currents, tuple(config["reference_window_s"]))
            noisy = jung_thdn_augment(
                time, currents, omega, level, rng, dc=dc,
                metadata=blocked.get(rid, ()),
                limiter=(config["jung_limiter"] == "blocked_phase"),
                quiet_ratio=config["jung_quiet_ratio"],
            )
        else:
            noisy = add_awgn(currents, level, rng)
        output[rid] = (time, noisy, omega)
    return output


def train(args):
    """一组源域训练；全程不接触目标数据。"""
    import numpy as np
    import torch
    from torch.nn import functional as F

    from jung_hang_optim_v6 import AdaMod

    config, ref, paper, provenance, raw, meta, labels, clean, val_labels, val = \
        source_data(args, args.method)
    method, seed = args.method, args.seed
    if method not in config["models"] or seed not in config["seeds"]:
        raise ValueError("method/seed not in frozen config")

    folder = args.output / f"models/{method}/seed{seed}"
    if folder.exists():
        raise FileExistsError(folder)

    ref.fix_randomness(seed)
    model = paper.get_model(method, config).cuda()
    optimizer = (AdaMod(model.parameters(), lr=config["learning_rate"],
                        weight_decay=config["weight_decay"])
                 if method == "jung"
                 else torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                       weight_decay=config["weight_decay"]))
    initial = ref.digest_parameters(model)

    states, history = {}, []
    for epoch in range(1, config["epochs_by_method"][method] + 1):
        modified = noise_records(raw, seed, epoch, method=method, meta=meta, config=config)
        noisy, noisy_positions = paper.build_inputs(method, meta, modified, config)
        np.testing.assert_array_equal(noisy_positions, np.arange(len(meta)))
        noise_digest = hashlib.sha256(np.ascontiguousarray(noisy).tobytes()).hexdigest()

        order = np.random.default_rng(seed + 901 + epoch).permutation(len(labels))
        model.train()
        losses, weighted = [], 0.0
        for start in range(0, len(order), config["batch_size"]):
            selected = order[start:start + config["batch_size"]]
            x_clean = torch.from_numpy(clean[selected]).cuda()
            x_noisy = torch.from_numpy(noisy[selected]).cuda()
            y = torch.from_numpy(labels[selected]).cuda()
            optimizer.zero_grad(set_to_none=True)
            n = len(selected)
            scores = model(torch.cat((x_clean, x_noisy), dim=0))
            loss = 0.5 * (F.cross_entropy(scores[:n], y) + F.cross_entropy(scores[n:], y))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all())
                       for p in model.parameters()):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            weighted += value * n

        scores, ces = {}, {}
        for name, values in val.items():
            predicted, ce = ref.predict(model, values, torch.device("cuda"), val_labels)
            scores[name] = ref.metrics(val_labels, predicted)["macro_f1"]
            ces[name] = ce
        row = {
            "epoch": epoch,
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_sample_weighted": weighted / len(labels),
            "optimizer_steps": len(losses),
            "clean_f1": scores["clean"],
            "clean_ce": ces["clean"],
            "noisy_f1": {str(s): scores[f"snr{s}"] for s in config["val_noise_db"]},
            "noisy_ce": {str(s): ces[f"snr{s}"] for s in config["val_noise_db"]},
            "noise_mean_f1": float(np.mean([scores[f"snr{s}"] for s in config["val_noise_db"]])),
            "noise_mean_ce": float(np.mean([ces[f"snr{s}"] for s in config["val_noise_db"]])),
            "training_noise_sha256": noise_digest,
            "sampling_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        }
        history.append(row)
        states[epoch] = ref.snapshot(model)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{method} seed{seed} epoch{epoch}/{config['epochs_by_method'][method]} "
                  f"loss={row['train_loss_mean']:.6f} noiseF1={row['noise_mean_f1']:.6f}",
                  flush=True)

    chosen = ref.select_source_checkpoint(history)
    folder.mkdir(parents=True)
    common = {
        "method": method, "seed": seed, "protocol": config["protocol"],
        "config": config, "source_split": provenance, "selected": chosen,
        "target_used_for_training_or_selection": False,
        "final_holdout_evaluated": False,
    }
    torch.save({**common, "state": states[chosen["epoch"]]}, folder / "best.pt")
    torch.save({**common, "state": states[config["epochs_by_method"][method]],
                "epoch": config["epochs_by_method"][method]}, folder / "last.pt")
    write(folder / "history.json", history)
    write(folder / "run.json", {
        **common,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_parameters_sha256": initial,
        "source_train_cycles": len(labels),
        "source_val_cycles": len(val_labels),
        "first_training_noise_sha256": history[0]["training_noise_sha256"],
    })


def target_root(args, snr):
    """返回某一档固定 Dev 数据的根目录；不生成任何目标噪声。"""
    _, _, _, _, _, manifest = setup(args)
    if snr in (4, -4, -8):
        return args.additional_root / "data"
    return Path(manifest["strong_root"] if snr in (0, 5) else manifest["extreme_root"])


def evaluate(args, replay=False):
    """在同一 2474 个 Dev 窗口上评价两个模型的两个种子。"""
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, recall_score,
    )

    config, ref, paper, _, _ = setup(args)[:5]
    models = {}
    for method in config["models"]:
        for seed in config["seeds"]:
            saved = torch.load(args.output / f"models/{method}/seed{seed}/best.pt",
                               map_location="cpu", weights_only=True)
            if saved["config"] != config:
                raise ValueError("checkpoint config mismatch")
            ref.fix_randomness(seed)
            model = paper.get_model(method, config).cuda().eval()
            model.load_state_dict(saved["state"], strict=True)
            models[method, seed] = (model, saved)
    before = {key: ref.snapshot(pair[0]) for key, pair in models.items()}

    count = 0
    for snr in config["target_conditions_db"]:
        raw, metadata = ref.data.target_dev_from_mat(snr, target_root(args, snr))
        if (len(raw), len(metadata)) != (config["target_records_expected"],
                                         config["target_windows_expected"]):
            raise ValueError("target Dev cohort changed")
        truth = np.asarray([row["label"] for row in metadata], dtype=np.int64)
        for method in config["models"]:
            values, positions = paper.build_inputs(method, metadata, raw, config)
            np.testing.assert_array_equal(positions, np.arange(len(metadata)))
            for seed in config["seeds"]:
                model, saved = models[method, seed]
                predicted, _ = ref.predict(model, values, torch.device("cuda"))
                score = ref.metrics(truth, predicted)
                path = args.output / f"evaluations/{method}/seed{seed}/target_dev_snr{snr}.json"
                if replay:
                    obj = read(path)
                    np.testing.assert_array_equal(obj["truth"], truth)
                    np.testing.assert_array_equal(obj["predicted"], predicted)
                    np.testing.assert_allclose(obj["metrics"]["macro_f1"],
                                               f1_score(truth, predicted, average="macro"),
                                               rtol=0, atol=1e-15)
                    np.testing.assert_allclose(obj["metrics"]["accuracy"],
                                               accuracy_score(truth, predicted),
                                               rtol=0, atol=1e-15)
                    np.testing.assert_allclose(obj["metrics"]["per_class_f1"],
                                               f1_score(truth, predicted, average=None),
                                               rtol=0, atol=1e-15)
                    np.testing.assert_allclose(obj["metrics"]["per_class_recall"],
                                               recall_score(truth, predicted, average=None),
                                               rtol=0, atol=1e-15)
                    np.testing.assert_array_equal(obj["metrics"]["confusion_matrix"],
                                                  confusion_matrix(truth, predicted,
                                                                   labels=list(range(6))))
                else:
                    if path.exists():
                        raise FileExistsError(path)
                    write(path, {
                        "method": method, "seed": seed, "condition": f"snr{snr}",
                        "metrics": score, "truth": truth.tolist(),
                        "predicted": predicted.tolist(), "windows": len(truth),
                        "record_families": len(raw), "config": config,
                        "source_split": saved["source_split"],
                        "selected_source_epoch": saved["selected"]["epoch"],
                        "target_used_for_training_or_selection": False,
                        "final_holdout_evaluated": False,
                    })
                count += 1
                print(f"{'REPLAY' if replay else 'DEV'} {method} seed{seed} "
                      f"{snr}dB Macro-F1={100 * score['macro_f1']:.3f}", flush=True)

    for key, (model, _) in models.items():
        if not all(torch.equal(v.detach().cpu(), before[key][k])
                   for k, v in model.state_dict().items()):
            raise ValueError("inference changed model state")

    if replay:
        rows = []
        for method in config["models"]:
            *_, val_labels, val = source_data(args, method)
            for seed in config["seeds"]:
                model, saved = models[method, seed]
                history = read(args.output / f"models/{method}/seed{seed}/history.json")
                chosen = ref.select_source_checkpoint(history)
                if chosen != saved["selected"]:
                    raise ValueError("selected checkpoint replay mismatch")
                if len(history) != config["epochs_by_method"][method]:
                    raise ValueError("epoch count mismatch")
                for name, values in val.items():
                    predicted, ce = ref.predict(model, values, torch.device("cuda"), val_labels)
                    expected_f1 = (chosen["clean_f1"] if name == "clean"
                                   else chosen["noisy_f1"][name[3:]])
                    expected_ce = (chosen["clean_ce"] if name == "clean"
                                   else chosen["noisy_ce"][name[3:]])
                    np.testing.assert_allclose(ref.metrics(val_labels, predicted)["macro_f1"],
                                               expected_f1, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(ce, expected_ce, rtol=0, atol=1e-9)
                    rows.append({"method": method, "seed": seed, "condition": name,
                                 "macro_f1": expected_f1, "ce": ce})
        ref.write_csv(args.output / "reports/SOURCE_VALIDATION_REPLAY.csv", rows)
        write(args.output / "VERIFICATION.json", {
            "status": "passed", "target_prediction_replays": count,
            "source_validation_replays": len(rows),
            "model_parameters_and_bn_unchanged": True,
            "final_holdout_evaluated": False,
        })


def preflight(args):
    """源域输入审计：轨迹图退化、THD、注噪覆盖率、健康尺度回退比例。"""
    import numpy as np

    from jung_hang_models_v6 import (
        BLOCKED_PHASES, fundamental_thd, record_reference,
    )

    config, ref, paper, split, provenance, _ = setup(args)
    (raw, meta, labels, _), (_, val_meta, val_labels, _, _) = split.source_views(args.source_root)
    truth = np.asarray(labels)
    val_truth = np.asarray(val_labels)

    report = {"protocol": config["protocol"], "source_split": provenance}
    for method in config["models"]:
        clean, positions = paper.build_inputs(method, meta, raw, config)
        np.testing.assert_array_equal(positions, np.arange(len(meta)))
        val_clean, _ = paper.build_inputs(method, val_meta, raw, config)
        entry = {"train_shape": list(clean.shape), "val_shape": list(val_clean.shape),
                 "train_cycles": len(meta), "val_cycles": len(val_meta)}
        if method == "hang":
            entry.update(paper.image_audit(clean, truth))
            entry["val_image_audit"] = paper.image_audit(val_clean, val_truth)
        else:
            entry["value_range"] = [float(clean.min()), float(clean.max())]
        report[method] = entry

    fallback, scales, thd, coverage = [], [], [], {}
    targets = [config["jung_thdn_percent"]] if isinstance(config["jung_thdn_percent"], int) \
        else list(config["jung_thdn_percent"])
    for target in targets:
        coverage[str(target)] = {"injected_channels": 0, "total_channels": 0}
    for rid in sorted(raw):
        time, currents, omega = raw[rid]
        dc, scale, used = record_reference(time, currents, tuple(config["reference_window_s"]))
        if not used:
            fallback.append(rid)
        scales.append(scale)
        fundamental, native = fundamental_thd(time, currents, omega, dc)
        thd.append(native)
        for target in targets:
            desired = target / 100.0
            sigma = fundamental * np.sqrt(np.maximum(desired ** 2 - native ** 2, 0.0))
            coverage[str(target)]["injected_channels"] += int((sigma > 0).sum())
            coverage[str(target)]["total_channels"] += int(len(sigma))
    thd = np.asarray(thd)
    report["jung_thdn"] = {
        "native_thd_percent": {
            "median": float(100 * np.median(thd)),
            "p95": float(100 * np.percentile(thd, 95)),
            "max": float(100 * thd.max()),
        },
        "nonzero_injection_fraction": {
            key: value["injected_channels"] / value["total_channels"]
            for key, value in coverage.items()
        },
    }
    report["hang_scale"] = {
        "median": float(np.median(scales)),
        "min": float(np.min(scales)),
        "max": float(np.max(scales)),
        "healthy_reference_fallback_records": len(fallback),
    }
    report["blocked_phases"] = {str(k): list(v) for k, v in BLOCKED_PHASES.items()}
    report["smoke"] = paper.smoke(config)
    write(args.output / "PREFLIGHT_PAPER_AUDIT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def summarize(args):
    """保存逐种子指标，并与已有十二模型七档结果客观合表。"""
    import csv
    import numpy as np

    config, ref, _, _, _ = setup(args)[:5]
    if read(args.output / "VERIFICATION.json")["status"] != "passed":
        raise ValueError("run verify before summarize")

    rows, classes, history_rows, selected = [], [], [], []
    for method in config["models"]:
        for seed in config["seeds"]:
            run = read(args.output / f"models/{method}/seed{seed}/run.json")
            selected.append({"method": method, "seed": seed,
                             "selected_epoch": run["selected"]["epoch"],
                             "parameters": run["parameters"]})
            for row in read(args.output / f"models/{method}/seed{seed}/history.json"):
                history_rows.append({"method": method, "seed": seed, **{
                    k: v for k, v in row.items() if not isinstance(v, dict)} })
            for snr in config["target_conditions_db"]:
                obj = read(args.output / f"evaluations/{method}/seed{seed}/target_dev_snr{snr}.json")
                s = obj["metrics"]
                rows.append({"method": method, "seed": seed, "snr_db": snr,
                             "macro_f1_percent": 100 * s["macro_f1"],
                             "accuracy_percent": 100 * s["accuracy"],
                             "fault_as_healthy": s["missed_fault_as_normal"],
                             "healthy_as_fault": s["normal_false_alarms"]})
                classes.extend({"method": method, "seed": seed, "snr_db": snr,
                                "label": label, "f1_percent": 100 * f1,
                                "recall_percent": 100 * recall}
                               for label, (f1, recall) in enumerate(
                                   zip(s["per_class_f1"], s["per_class_recall"])))

    previous_path = args.previous_output / "results/reports/PER_SEED_RESULTS.csv"
    with previous_path.open(encoding="utf8") as stream:
        previous = list(csv.DictReader(stream))
    if len(previous) != 168:
        raise ValueError("previous twelve-model table incomplete")
    combined = previous + rows

    aggregate = []
    for method in dict.fromkeys(r["method"] for r in combined):
        for snr in config["target_conditions_db"]:
            subset = [r for r in combined
                      if r["method"] == method and int(r["snr_db"]) == snr]
            values = [float(r["macro_f1_percent"]) for r in subset]
            if len(values) != 2:
                raise ValueError("seeds not paired")
            aggregate.append({"method": method, "snr_db": snr,
                              "macro_f1_mean": float(np.mean(values)),
                              "macro_f1_sd": float(np.std(values, ddof=1))})

    for name, data in (("PER_SEED_RESULTS.csv", rows),
                       ("PER_CLASS_PER_SEED.csv", classes),
                       ("TRAINING_HISTORY.csv", history_rows),
                       ("SELECTED_CHECKPOINTS.csv", selected),
                       ("ALL_14_MODELS_PER_SEED.csv", combined),
                       ("ALL_14_MODELS_SUMMARY.csv", aggregate)):
        ref.write_csv(args.output / "reports" / name, data)
    print(f"SUMMARY: {len(rows)} new tests, {len(combined)} combined tests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "evaluate", "verify", "summarize"))
    parser.add_argument("--method", choices=("jung", "hang"))
    parser.add_argument("--seed", type=int, choices=(42, 44))
    parser.add_argument("--config", type=Path, default=CONFIG)
    for name in ("reference-root", "source-root", "additional-root", "previous-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    start = time.time()
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "train":
        if args.method is None or args.seed is None:
            parser.error("train requires --method and --seed")
        train(args)
    elif args.stage == "evaluate":
        evaluate(args)
    elif args.stage == "verify":
        evaluate(args, replay=True)
    else:
        summarize(args)
    print(f"{args.stage} finished in {time.time() - start:.3f}s", flush=True)


if __name__ == "__main__":
    main()
