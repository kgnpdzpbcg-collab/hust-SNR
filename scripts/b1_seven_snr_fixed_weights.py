"""八模型七档固定权重评价：预检→补42组→回放112组→汇总。

模型从既有不可变训练快照加载，网络、训练配置、best轮次与数据均不修改。
旧70份预测逐字节引用；新增42份另存。本脚本不创建优化器、不执行训练。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CONFIG = Path(__file__).resolve().parents[1] / "configs/b1_seven_snr_fixed_weights_v1.json"
MODELS = ("b1", "a1_fixed_lp4_dual", "a2_fixed_noise_proxy", "a3_no_interphase_descriptors",
          "a4_md_only", "a5_no_strength_loss", "xu_full", "b0")
SEEDS = (42, 44)
CONDITIONS = (5, 4, 0, -4, -5, -8, -10)
EXTRA = (4, -4, -8)


def read(path):
    """读取固定UTF-8元数据；不从目标波形推断标签或样本边界。"""
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    """仅向调用方指定的新输出目录写JSON，使用LF以便跨平台核查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf8", newline="\n")


def sha(path):
    """记录原文件字节指纹，包括模型参数、BN缓冲区和A2固定代理。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    """完整保留实际种子，不按成绩过滤；CSV数值保留原浮点精度。"""
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configuration():
    """只允许已批准的八模型、两种子、七档协议，防止混入其他任务。"""
    c = read(CONFIG)
    if (c["models"], c["seeds"], c["conditions_db"], c["additional_conditions_db"]) != (
            list(MODELS), list(SEEDS), list(CONDITIONS), list(EXTRA)):
        raise ValueError("模型、种子或噪声档位偏离冻结协议")
    if c["retrained"] or c["target_used_for_training_or_selection"] or c["final_holdout_evaluated"]:
        raise ValueError("固定权重协议不允许训练、目标选模或最终保留集评价")
    return c


def setup(args):
    """先核查共享算法原字节，再从消融快照统一导入原模型和读取器。"""
    c = configuration()
    baseline = read(args.baseline_root / "DEPLOYMENT_MANIFEST.json")
    ablation = read(args.ablation_root / "DEPLOYMENT_MANIFEST.json")
    if baseline["code_commit"] != c["baseline_training_commit"] or ablation["commit"] != c["ablation_training_commit"]:
        raise ValueError("训练快照版本不匹配")
    for name, expected in ablation["code_sha256"].items():
        if sha(args.ablation_root / "code" / name) != expected:
            raise ValueError(f"消融快照改变：{name}")
    for path in (args.baseline_root / "code/scripts").glob("*.py"):
        if sha(path) != sha(args.ablation_root / "code/scripts" / path.name):
            raise ValueError(f"原基准与消融共用代码不同：{path.name}")
    sys.path.insert(0, str(args.ablation_root / "code/scripts"))
    import source_cycle_benchmark as original
    import b1_paper_ablations as ablation_runner
    from source_cycle_split import source_provenance
    from prepare import OUT
    provenance = source_provenance(args.source_root)
    if provenance["manifest_sha256"] != c["source_split_manifest_sha256"]:
        raise ValueError("源域划分版本改变")
    metadata = read(OUT / "target_dev_clean_windows.json")
    if len(metadata) != c["cycles_per_condition"] or len({r["record_id"] for r in metadata}) != c["record_families"]:
        raise ValueError("固定Dev队列改变")
    return c, original, ablation_runner, provenance, metadata


def checkpoint_folder(args, method, seed):
    """完整B1、Xu、B0来自4728e9b；五个消融来自d9947f0。"""
    root = args.ablation_root if method.startswith("a") else args.baseline_root
    return root / "results/models" / method / f"seed{seed}"


def prior_prediction(args, method, seed, snr):
    """返回可复用的旧文件；None明确表示本轮需要新增评价。"""
    if snr in EXTRA:
        if method != "b1":
            return None
        root = args.additional_root / "evaluation"
    else:
        root = (args.ablation_root if method.startswith("a") else args.baseline_root) / "results"
    return root / f"evaluations/{method}/seed{seed}/target_dev_snr{snr}.json"


def condition_root(args, snr):
    """三个新增档位读既有派生目录，原四档仍读其原始输入目录。"""
    if snr in EXTRA:
        return args.additional_root / "data"
    m = read(args.ablation_root / "DEPLOYMENT_MANIFEST.json")
    return Path(m["strong_root"] if snr in (0, 5) else m["extreme_root"])


def mat_path(root, snr, record_id):
    """与原读取器使用相同候选目录；发现重复候选时显式停止。"""
    candidates = [root / f"snr{snr}" / f"{record_id}.mat", root / f"SNR_{snr}" / f"{record_id}.mat",
                  root / str(snr) / f"{record_id}.mat"]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"数据文件缺失或重复：{record_id}/{snr}")
    return found[0]


def assert_protected(args):
    """各阶段前后核对原输入与旧结果，禁止静默覆盖历史实验。"""
    protected = read(args.output / "PROTECTED_FILES_SHA256.json")
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError(f"受保护文件改变：{path}")
    return protected


def load_models(args, original, ablation, provenance):
    """严格加载完整state_dict，使用原选中轮次，不校准或微调目标模型。"""
    import torch
    from v1_noise_consistency import fix_randomness
    models, runs = {}, {}
    for method in MODELS:
        for seed in SEEDS:
            folder = checkpoint_folder(args, method, seed)
            saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            run = read(folder / "run.json")
            expected_config = ablation.configuration() if method.startswith("a") else original.configuration()
            if saved["config"] != expected_config or run["config"] != expected_config:
                raise ValueError("权重配置与原训练快照不一致")
            if saved["source_split"] != provenance or saved["selected"] != run["selected"]:
                raise ValueError("权重划分或最佳轮次改变")
            if (saved["method"], saved["seed"]) != (method, seed):
                raise ValueError("模型身份改变")
            if run["target_used_for_training_or_selection"] or saved["final_holdout_evaluated"]:
                raise ValueError("目标域使用标记不符合本协议")
            fix_randomness(seed)
            model = (ablation.build_model(method) if method.startswith("a") else original.build_model(method)).cuda().eval()
            model.load_state_dict(saved["state"], strict=True)
            models[method, seed], runs[method, seed] = model, run
    return models, runs


def preflight(args):
    """核查264份既有派生MAT及所有固定权重，写入新目录后才允许补测。"""
    import numpy as np
    import torch
    from scipy.io import loadmat
    c, original, ablation, provenance, metadata = setup(args)
    if args.output.exists():
        raise FileExistsError("拒绝覆盖本轮输出目录")
    for path in (args.ablation_root / "VERIFICATION.json", args.additional_root / "VERIFICATION.json",
                 args.baseline_root / "results/reports/VERIFICATION.json"):
        if read(path)["status"] != "passed":
            raise ValueError(f"既有实验未核验：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("本轮复用服务器CUDA环境")
    ids = sorted({r["record_id"] for r in metadata})
    truth = np.asarray([r["label"] for r in metadata])
    if np.bincount(truth, minlength=6).tolist() != c["class_counts"]:
        raise ValueError("测试类别数量改变")
    generation = read(args.additional_root / "data/GENERATION_MANIFEST.json")
    for path, expected in generation["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"作者参考数据改变：{path}")
    max_snr_error = 0.0
    clean_root = args.baseline_root / "code/data/target_dev/clean"
    ref_root = Path(read(args.baseline_root / "DEPLOYMENT_MANIFEST.json")["snr_root"]) / "SNR_10"
    for entry in generation["files"]:
        path = args.additional_root / "data" / entry["file"]
        if sha(path) != entry["file_sha256"]:
            raise ValueError(f"既有派生数据改变：{path}")
        value = loadmat(path, variable_names=["I_noisy"])["I_noisy"]
        clean = loadmat(clean_root / f"{entry['record_id']}.mat", variable_names=["name"])["name"]
        ref = loadmat(ref_root / f"{entry['record_id']}.mat", variable_names=["I_noisy"])["I_noisy"]
        if value.shape != (10031, 7) or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("MAT格式改变")
        np.testing.assert_array_equal(value[:, [0, 6]], clean[:, [0, 6]])
        noise = value[:, 1:6] - clean[:, 1:6]
        np.testing.assert_allclose(noise, (ref[:, 1:6] - clean[:, 1:6]) * 10**((10-entry["snr_db"])/20), rtol=1e-12, atol=1e-12)
        snr = 10*np.log10(np.mean(clean[:, 1:6]**2, axis=0)/np.mean(noise**2, axis=0))
        max_snr_error = max(max_snr_error, float(np.max(np.abs(snr-entry["snr_db"]))))
    if len(generation["files"]) != 264 or max_snr_error > 1e-6:
        raise ValueError("派生数据数量或实际SNR不符合协议")
    for snr in EXTRA:
        if sorted(p.stem for p in (args.additional_root / f"data/SNR_{snr}").glob("*.mat")) != ids:
            raise ValueError("派生数据编号改变")
    paths = list(args.source_root.rglob("*.json")) + list((args.source_root / "raw_source").glob("*.mat"))
    paths += list((args.ablation_root / "code/scripts").glob("*.py"))
    paths += [Path(path) for path in generation["input_sha256"]]
    paths += [mat_path(condition_root(args, snr), snr, rid) for snr in CONDITIONS for rid in ids]
    # 连同未参与本次推理的历史种子一起核验，确保既有104份权重未被改动。
    for weight_manifest in (args.baseline_root / "results/reports/WEIGHTS_SHA256.json",
                            args.ablation_root / "results/reports/WEIGHTS_SHA256.json"):
        for path, expected in read(weight_manifest).items():
            if sha(path) != expected:
                raise ValueError(f"历史权重发生变化：{path}")
            paths.append(Path(path))
    prior_count = 0
    for method in MODELS:
        for seed in SEEDS:
            paths += [checkpoint_folder(args, method, seed)/name for name in ("best.pt", "last.pt", "run.json", "history.json")]
            for snr in CONDITIONS:
                prior = prior_prediction(args, method, seed, snr)
                if prior is not None:
                    paths.append(prior)
                    prior_count += 1
    if prior_count != 70:
        raise ValueError("历史评价数量不符")
    models, runs = load_models(args, original, ablation, provenance)
    # 用真实源验证周期检查Xu字典接口及其他七模型张量接口，保持eval状态。
    (_, val), _ = original.checked_source(args.source_root, original.configuration())
    for key, model in models.items():
        with torch.inference_mode():
            output = original.logits(model, torch.from_numpy(val[3][:2]).cuda())
        if output.shape != (2, 6) or not bool(torch.isfinite(output).all()):
            raise ValueError(f"模型真实样本前向失败：{key}")
    args.output.mkdir(parents=True)
    write(args.output / "PROTECTED_FILES_SHA256.json", {str(p): sha(p) for p in sorted(set(paths))})
    write(args.output / "PREFLIGHT.json", {"status": "passed", "config": c, "source_split": provenance,
        "models_loaded_strictly": 16, "source_cuda_forward_checks": 16, "existing_predictions": prior_count,
        "existing_additional_mat_checked": 264, "max_whole_record_per_phase_snr_error_db": max_snr_error,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "interpreter": sys.executable,
        "retrained": False, "final_holdout_evaluated": False})
    print("PREFLIGHT PASSED: 16 weights, 264 MAT, 70 prior predictions", flush=True)


def evaluate(args):
    """旧70份JSON原样复制；五消融、Xu和B0仅新增三档42组预测。"""
    import numpy as np
    import torch
    c, original, ablation, provenance, metadata = setup(args)
    if read(args.output / "PREFLIGHT.json")["status"] != "passed":
        raise ValueError("必须先通过预检")
    assert_protected(args)
    folder = args.output / "results/evaluations"
    if folder.exists():
        raise FileExistsError("拒绝覆盖本轮评价")
    models, runs = load_models(args, original, ablation, provenance)
    sources, reused = [], 0
    for method in MODELS:
        for seed in SEEDS:
            for snr in CONDITIONS:
                prior = prior_prediction(args, method, seed, snr)
                destination = folder / f"{method}/seed{seed}/target_dev_snr{snr}.json"
                if prior is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(prior, destination)
                    sources.append({"method": method, "seed": seed, "snr_db": snr, "kind": "reused",
                        "source_path": str(prior), "source_sha256": sha(prior)})
                    reused += 1
    added = 0
    truth = np.asarray([r["label"] for r in metadata], dtype=np.int64)
    for snr in EXTRA:
        raw, actual = original.data.target_dev_from_mat(snr, condition_root(args, snr))
        if actual != metadata or len(raw) != 88:
            raise ValueError("测试周期或文件编号改变")
        x = original.data.raw_cycles(raw, metadata)
        for (method, seed), model in models.items():
            if method == "b1":
                continue
            with torch.inference_mode():
                predicted, _ = original.predict(model, x, torch.device("cuda"))
            destination = folder / f"{method}/seed{seed}/target_dev_snr{snr}.json"
            write(destination, {"protocol": c["protocol"], "method": method, "seed": seed, "condition": f"snr{snr}",
                "source_split": provenance, "source_only_selection": runs[method, seed]["selected"],
                "weights_sha256": sha(checkpoint_folder(args, method, seed)/"best.pt"),
                "record_families": 88, "windows": 2474, "truth": truth.tolist(), "predicted": predicted.tolist(),
                "metrics": original.metrics(truth, predicted), "target_used_for_training_or_selection": False,
                "final_holdout_evaluated": False})
            sources.append({"method": method, "seed": seed, "snr_db": snr, "kind": "new",
                "source_path": str(destination), "source_sha256": sha(destination)})
            added += 1
            print(f"NEW {method}/seed{seed}/{snr}dB F1={100*original.metrics(truth,predicted)['macro_f1']:.4f}", flush=True)
    if (added, reused) != (42, 70):
        raise ValueError("新增或引用数量不符")
    assert_protected(args)
    write(args.output / "RESULT_PROVENANCE.json", sources)
    write(args.output / "EVALUATION.json", {"status": "completed", "new": added, "reused": reused, "total": added+reused})


def verify(args):
    """重新加载全部16份best，回放112份预测及96组源验证，独立复算指标。"""
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
    c, original, ablation, provenance, metadata = setup(args)
    protected = assert_protected(args)
    if read(args.output / "EVALUATION.json")["total"] != 112:
        raise ValueError("评价未齐全")
    models, runs = load_models(args, original, ablation, provenance)
    before = {key: original.snapshot(model) for key, model in models.items()}
    source_rows = []
    (_, val), _ = original.checked_source(args.source_root, original.configuration())
    _, _, vy, clean, noisy = val
    for key, model in models.items():
        selected = runs[key]["selected"]
        for name, x in [("clean", clean), *[(f"snr{s}", x) for s,x in noisy.items()]]:
            predicted, ce = original.predict(model, x, torch.device("cuda"), vy)
            expected_f1 = selected["clean_f1"] if name == "clean" else selected["noisy_f1"][name[3:]]
            expected_ce = selected["clean_ce"] if name == "clean" else selected["noisy_ce"][name[3:]]
            score = f1_score(vy, predicted, average="macro")
            np.testing.assert_allclose(score, expected_f1, rtol=0, atol=1e-15)
            np.testing.assert_allclose(ce, expected_ce, rtol=0, atol=1e-9)
            source_rows.append({"method":key[0],"seed":key[1],"condition":name,"macro_f1":score,"ce":ce})
    truth = np.asarray([r["label"] for r in metadata], dtype=np.int64)
    checked = 0
    for snr in CONDITIONS:
        raw, actual = original.data.target_dev_from_mat(snr, condition_root(args, snr))
        if actual != metadata:
            raise ValueError("样本顺序改变")
        x = original.data.raw_cycles(raw, metadata)
        for (method, seed), model in models.items():
            obj = read(args.output / f"results/evaluations/{method}/seed{seed}/target_dev_snr{snr}.json")
            if (obj["method"],obj["seed"],obj["condition"],obj["windows"],obj["record_families"]) != (method,seed,f"snr{snr}",2474,88):
                raise ValueError("预测身份或样本数改变")
            np.testing.assert_array_equal(obj["truth"], truth)
            predicted, _ = original.predict(model, x, torch.device("cuda"))
            np.testing.assert_array_equal(predicted, obj["predicted"])
            score = obj["metrics"]
            np.testing.assert_allclose(score["macro_f1"],f1_score(truth,predicted,average="macro"),rtol=0,atol=1e-15)
            np.testing.assert_allclose(score["accuracy"],accuracy_score(truth,predicted),rtol=0,atol=1e-15)
            np.testing.assert_allclose(score["per_class_f1"],f1_score(truth,predicted,average=None),rtol=0,atol=1e-15)
            np.testing.assert_allclose(score["per_class_recall"],recall_score(truth,predicted,average=None),rtol=0,atol=1e-15)
            cm = confusion_matrix(truth,predicted,labels=list(range(6)))
            np.testing.assert_array_equal(score["confusion_matrix"],cm)
            if score["missed_fault_as_normal"] != int(cm[1:,0].sum()) or score["normal_false_alarms"] != int(cm[0,1:].sum()):
                raise ValueError("漏报或误报计数不符")
            if obj.get("source_split",provenance) != provenance or obj["final_holdout_evaluated"]:
                raise ValueError("测试数据版本或最终保留集标记不符")
            checked += 1
    for key, model in models.items():
        for name, value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(),before[key][name]):
                raise ValueError("推理改变模型状态")
    for row in read(args.output / "RESULT_PROVENANCE.json"):
        if sha(row["source_path"]) != row["source_sha256"]:
            raise ValueError("结果来源文件改变")
    assert_protected(args)
    reports = args.output / "results/reports"
    reports.mkdir(parents=True)
    save_csv(reports / "SOURCE_VALIDATION_REPLAY.csv",source_rows)
    write(args.output / "VERIFICATION.json", {"status":"passed","target_predictions_replayed_exactly":checked,
        "source_validation_replays":len(source_rows),"independent_sklearn_metric_checks":checked,
        "protected_files_unchanged":len(protected),"new_evaluations":42,"reused_evaluations":70,
        "fixed_best_weights":16,"model_parameters_and_bn_unchanged":True,"retrained":False,
        "final_holdout_evaluated":False,"verification_scope":"fixed-weight replay; no second training"})
    print(f"VERIFIED {checked} target predictions; {len(source_rows)} source validation sets",flush=True)


def summarize(args):
    """按批准顺序保存七档、两种子、各类指标及同种子Full B1配对差。"""
    import numpy as np
    if read(args.output / "VERIFICATION.json")["status"] != "passed":
        raise ValueError("预测核验通过后才可汇总")
    rows, classes, summary, pairs, selected = [], [], [], [], []
    for method in MODELS:
        for seed in SEEDS:
            run = read(checkpoint_folder(args,method,seed)/"run.json")
            selected.append({"method":method,"seed":seed,"selected_epoch":run["selected"]["epoch"],
                "best_weight_sha256":sha(checkpoint_folder(args,method,seed)/"best.pt"),
                "source_train_cycles":run["source_train_cycles"],"source_val_cycles":run["source_val_cycles"]})
            for snr in CONDITIONS:
                obj = read(args.output / f"results/evaluations/{method}/seed{seed}/target_dev_snr{snr}.json")
                score = obj["metrics"]
                rows.append({"method":method,"seed":seed,"condition":f"snr{snr}","snr_db":snr,
                    "macro_f1_percent":100*score["macro_f1"],"accuracy_percent":100*score["accuracy"],
                    "fault_as_healthy":score["missed_fault_as_normal"],"healthy_as_fault":score["normal_false_alarms"]})
                classes.extend({"method":method,"seed":seed,"snr_db":snr,"label":label,
                    "f1_percent":100*f1,"recall_percent":100*recall,"support":configuration()["class_counts"][label]}
                    for label,(f1,recall) in enumerate(zip(score["per_class_f1"],score["per_class_recall"])))
        for snr in CONDITIONS:
            subset = [r for r in rows if r["method"]==method and r["snr_db"]==snr]
            item = {"method":method,"snr_db":snr,"seeds":len(subset)}
            for metric in ("macro_f1","accuracy"):
                values=[r[metric+"_percent"] for r in subset]
                item[metric+"_mean"]=float(np.mean(values))
                item[metric+"_sd"]=float(np.std(values,ddof=1))
            summary.append(item)
    for row in rows:
        if row["method"]!="b1":
            full=next(r for r in rows if r["method"]=="b1" and r["seed"]==row["seed"] and r["snr_db"]==row["snr_db"])
            pairs.append({"method":row["method"],"seed":row["seed"],"snr_db":row["snr_db"],
                "full_b1_macro_f1":full["macro_f1_percent"],"model_macro_f1":row["macro_f1_percent"],
                "full_minus_model_pp":full["macro_f1_percent"]-row["macro_f1_percent"]})
    if (len(rows),len(classes),len(summary),len(pairs))!=(112,672,56,98):
        raise ValueError("汇总表数量不符")
    reports=args.output / "results/reports"
    for filename,values in (("PER_SEED_RESULTS.csv",rows),("PER_CLASS_PER_SEED.csv",classes),("SUMMARY.csv",summary),
                            ("PAIRED_FULL_B1_MINUS_MODELS.csv",pairs),("SELECTED_CHECKPOINTS.csv",selected)):
        save_csv(reports/filename,values)
    write(reports/"SUMMARY.json",{"config":configuration(),"summary":summary,"retrained":False,"final_holdout_evaluated":False})
    print("SUMMARY: 112 seed results, 672 class results, 56 means, 98 paired differences",flush=True)


def main():
    """阶段失败立即返回非零；不自动重试，不覆盖此前同名输出。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("preflight","evaluate","verify","summarize"))
    for name in ("baseline-root","ablation-root","additional-root","source-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    started=time.time()
    {"preflight":preflight,"evaluate":evaluate,"verify":verify,"summarize":summarize}[args.stage](args)
    print(f"{args.stage} finished in {time.time()-started:.3f}s",flush=True)


if __name__=="__main__":
    main()
