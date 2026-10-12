"""服务器V1运行与核验：调用冻结分支原函数，复用原环境，隔离输出。

三种方法各跑42/43/44三个种子；最多三个任务并发。训练、损失、
预处理及选模均使用原分支实现。这里只负责输入核对、调度和客观记录。
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import csv
import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
OLD = ROOT.parents[2]
OUT = ROOT / "results"
SNR = ROOT / "target_snr"
sys.path.insert(0, str(CODE / "scripts"))
import numpy as np
import torch
import v1_noise_consistency as v1
from prepare import preprocess, OUT as GATE0, read

METHODS = ["baseline", "augmentation", "consistency"]
SEEDS = [42, 43, 44]
CONDITIONS = [None, 10, 15, 20, 25, 30, 35, 40]


def save(path, value):
    """保存UTF-8客观记录；不往旧实验目录写文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    """按原文件字节计算SHA256，检查输入和冻结权重是否变化。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def status(stage, **details):
    """更新可轮询的状态；UTC时间用于跨主机追踪，不影响随机种子。"""
    save(ROOT / "STATUS.json", {"stage": stage,
         "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **details})
    print(stage, details, flush=True)


def old_fingerprints():
    """保存旧配置、代码、Gate0数组和权重哈希，完成后确认未改动。"""
    files = [p for p in (OLD / "outputs/gate0_v1").rglob("*")
             if p.is_file() and p.suffix in {".json", ".npz", ".pt"}]
    files += list((OLD / "configs").glob("*.json"))
    files += [OLD / "scripts" / name for name in ["prepare.py", "train.py", "verify.py"]]
    return {str(p.relative_to(OLD)): sha(p) for p in files}


def preflight():
    """核对冻结输入，运行分支测试，并缓存只读Dev张量供独立复评。

    缓存仅包含固定Dev；不加载Holdout。Clean/10dB与Gate0输入逐位比较，
    新SNR使用作者文件。缓存不会替代分支自身的正式evaluate入口。
    """
    assert GATE0.resolve() == (OLD / "outputs/gate0_v1").resolve()
    assert torch.cuda.is_available(), "官方运行需要原GPU，不能静默退到CPU"
    manifest = read(CODE / "configs/TRANSFER_MANIFEST_gate0_v1.json")
    for name, item in manifest["files"].items():
        path = OLD / name
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], name
    for name in ["splits_gate0_v1.json", "protocol_gate0_v1.json", "TRANSFER_MANIFEST_gate0_v1.json"]:
        assert sha(CODE / "configs" / name) == sha(OLD / "configs" / name), name
    split = read(CODE / "configs/splits_gate0_v1.json")
    assert set(split["target_dev_records"]).isdisjoint(split["target_holdout_records"])
    assert not (CODE / "data/target_holdout").exists()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for label, args in [("unit_tests", ["-m", "pytest", "-q", "tests/test_v1_consistency.py"]),
                        ("smoke", ["scripts/v1_noise_consistency.py", "smoke"])]:
        with (ROOT / f"{label}.log").open("w") as log:
            subprocess.run([sys.executable, *args], cwd=CODE, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    meta = read(GATE0 / "target_dev_clean_windows.json")
    assert len(meta) == 2474 and {r["record_id"] for r in meta} == set(split["target_dev_records"])
    assert meta == read(GATE0 / "target_dev_snr10_windows.json")
    prepared = {}
    for snr in CONDITIONS:
        condition = "clean" if snr is None else f"snr{snr}"
        rows = {rid: v1.target_condition_record(rid, snr, SNR) for rid in split["target_dev_records"]}
        values = [preprocess(rows[r["record_id"]][0], rows[r["record_id"]][1], r["start"], r["end"])[0]
                  for r in meta]
        x = np.stack(values).astype(np.float32)
        y = np.asarray([r["label"] for r in meta], dtype=np.int64)
        if condition in ["clean", "snr10"]:
            with np.load(GATE0 / f"target_dev_{condition}.npz") as f:
                assert np.array_equal(x, f["x"]) and np.array_equal(y, f["y"]), condition
        np.savez_compressed(ROOT / f"dev_{condition}.npz", x=x, y=y)
        prepared[condition] = {"families": 88, "windows": len(y),
                              "tensor_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()}
    tr_meta = read(GATE0 / "source_train_windows.json")
    raw = v1.load_source_raw(r["record_id"] for r in tr_meta)
    matched = {}
    for seed in SEEDS:
        first = v1.fresh_train_noise(raw, tr_meta, 1, seed)
        second = v1.fresh_train_noise(raw, tr_meta, 1, seed)
        assert np.array_equal(first, second)
        matched[str(seed)] = hashlib.sha256(first.tobytes()).hexdigest()
    save(OUT / "PREFLIGHT.json", {"status": "passed", "unit_tests": 6,
         "code_commit": read(ROOT / "DEPLOYMENT_MANIFEST.json")["commit"],
         "source_input_files_hash_checked": len(manifest["files"]),
         "prepared_dev_conditions": prepared, "first_epoch_paired_noise_sha256": matched,
         "holdout_present": False, "clean_10db_equal_gate0_inputs": True})


def worker(task):
    """一个子进程负责一个方法/种子；调用原训练或评估，不改算法。

    评估前复用原fix_randomness，使FP32、TF32及确定性设置与训练一致。
    各任务单独保存日志；失败直接上报，不自动重训或覆盖现有输出。
    """
    stage, method, seed = task
    log_path = ROOT / f"{stage}_{method}_seed{seed}.log"
    started = time.perf_counter()
    with log_path.open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        if stage == "train":
            v1.train(method, seed, OUT, lam=0.1, epochs=50, batch_size=32)
        else:
            v1.fix_randomness(seed)
            v1.evaluate(method, seed, OUT, snr_root=SNR, conditions=CONDITIONS)
    return {"stage": stage, "method": method, "seed": seed,
            "wall_seconds": time.perf_counter() - started, "exit_status": "completed"}


def run_pool(stage):
    """最多三个并发任务，保持方法/种子独立，并记录完成数量。"""
    started = time.perf_counter()
    done = []
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker, (stage, m, s)) for m in METHODS for s in SEEDS]
        for future in as_completed(futures):
            done.append(future.result())
            status(stage, completed=len(done), total=9, last=done[-1])
    save(OUT / f"{stage.upper()}_EXECUTION.json",
         {"status": "completed", "max_workers": 3,
          "wall_seconds": time.perf_counter() - started, "tasks": done})


def freeze_weights():
    """先核对源域选模规则，再锁定全部九个权重；之后才进行目标推理。"""
    frozen = {}
    initial = {}
    for method in METHODS:
        for seed in SEEDS:
            folder = OUT / "models" / method / f"seed{seed}"
            run = read(folder / "run.json")
            history = read(folder / "history.json")
            assert len(history) == 50 and run["chosen"] == v1.score_checkpoints(history)
            assert run["target_used_in_training_or_selection"] is False
            initial.setdefault(seed, []).append(run["initial_param_sha256"])
            name = str((folder / "best.pt").relative_to(OUT))
            frozen[name] = {"sha256": sha(folder / "best.pt"), "selected": run["chosen"]}
    assert all(len(set(values)) == 1 for values in initial.values())
    save(OUT / "FROZEN_CHECKPOINTS.json", {"models": frozen, "initialization_matches_per_seed": True})
    return frozen


def verify_and_report(frozen, before):
    """独立重算72份指标，并重载九个权重逐条核对预测，输出客观表格。"""
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            v1.fix_randomness(seed)
            model = v1.CNN().cuda()
            checkpoint = OUT / "models" / method / f"seed{seed}/best.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state"])
            before_bn = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for snr in CONDITIONS:
                condition = "clean" if snr is None else f"snr{snr}"
                saved = read(OUT / "evaluations" / method / f"seed{seed}/target_dev_{condition}.json")
                with np.load(ROOT / f"dev_{condition}.npz") as f:
                    x, y = f["x"], f["y"]
                pred = np.asarray(saved["predicted"])
                assert np.array_equal(y, saved["truth"]) and len(pred) == 2474
                assert np.array_equal(pred, v1.predictions(model, x, "cuda"))
                m = saved["metrics"]
                cm = confusion_matrix(y, pred, labels=list(range(6)))
                assert np.array_equal(cm, m["confusion_matrix"])
                for key, value in [("macro_f1", f1_score(y, pred, average="macro")),
                                   ("accuracy", accuracy_score(y, pred)),
                                   ("balanced_accuracy", balanced_accuracy_score(y, pred))]:
                    assert np.isclose(value, m[key], rtol=0, atol=1e-12), (method, seed, condition, key)
                far = float(np.mean(pred[y == 0] != 0))
                assert far == m["normal_false_alarm_rate"]
                assert int(np.sum((y != 0) & (pred == 0))) == m["missed_fault_as_normal"]
                assert saved["final_holdout_evaluated"] is False
                rows.append({"method": method, "seed": seed, "condition": condition,
                             **{k + "_percent": m[k] * 100 for k in
                                ["macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"]}})
            assert all(torch.equal(before_bn[k], v.detach().cpu()) for k, v in model.state_dict().items())
    assert len(rows) == 72
    for name, info in frozen.items():
        assert sha(OUT / name) == info["sha256"]
    assert before == old_fingerprints(), "旧Gate0文件发生变化"
    aggregates = []
    for method in METHODS:
        for condition in ["clean", *[f"snr{s}" for s in CONDITIONS[1:]]]:
            subset = [r for r in rows if r["method"] == method and r["condition"] == condition]
            item = {"method": method, "condition": condition, "seeds": SEEDS}
            for key in [k for k in rows[0] if k.endswith("_percent")]:
                values = [r[key] for r in subset]
                item[key + "_mean"] = float(np.mean(values))
                item[key + "_sample_sd"] = float(np.std(values, ddof=1))
            aggregates.append(item)
    contrasts = []
    for condition in ["clean", *[f"snr{s}" for s in CONDITIONS[1:]]]:
        for newer, older in [("augmentation", "baseline"), ("consistency", "augmentation")]:
            delta = []
            for seed in SEEDS:
                lookup = {(r["method"], r["seed"]): r for r in rows if r["condition"] == condition}
                delta.append(lookup[(newer, seed)]["macro_f1_percent"] - lookup[(older, seed)]["macro_f1_percent"])
            contrasts.append({"condition": condition, "difference": newer + "_minus_" + older,
                              "per_seed_delta_pp": dict(zip(map(str, SEEDS), delta)),
                              "mean_delta_pp": float(np.mean(delta)),
                              "sample_sd_delta_pp": float(np.std(delta, ddof=1))})
    # 汇总文件由最新分支原summarize生成；独立计算只核对，不覆盖其结构。
    official = v1.summarize(OUT, methods=METHODS, seeds=SEEDS)
    for a in aggregates:
        item = next(v for v in official["aggregates"] if v["method"] == a["method"] and v["condition"] == a["condition"])
        for metric in ["macro_f1", "accuracy", "balanced_accuracy", "normal_false_alarm_rate"]:
            assert np.isclose(a[metric + "_percent_mean"], item[metric + "_mean_percent"], atol=1e-12, rtol=0)
            assert np.isclose(a[metric + "_percent_sample_sd"], item[metric + "_sd_percent"], atol=1e-12, rtol=0)
    for item in official["paired_consistency_minus_augmentation"]:
        own = next(v for v in contrasts if v["difference"] == "consistency_minus_augmentation" and v["condition"] == item["condition"])
        assert np.allclose(list(own["per_seed_delta_pp"].values()), item["m2_minus_m1_macro_f1_pp"], atol=1e-12, rtol=0)
    save(OUT / "INDEPENDENT_RECOMPUTATION.json", {"status": "passed", "aggregates": aggregates,
         "paired_contrasts": contrasts, "official_summary_equal": True})
    save(OUT / "VERIFICATION.json", {"status": "passed", "independent_metric_checks": 72,
         "checkpoint_reload_prediction_checks": 72, "model_bn_unchanged": True,
         "old_gate0_files_unchanged": True, "old_file_hash_count": len(before),
         "checkpoints_unchanged_after_target_evaluation": True, "final_holdout_evaluated": False})
    lines = ["# V1服务器实验记录", "", "提交：`69e2926ebd09f0a9b8fe8dfdebfd15aa5a7a5481`。",
             "三种方法使用分支原代码、50轮、种子42/43/44及源域含噪验证集统一选模。",
             "固定目标Dev为88条基础记录、每条件2474个周期。347条Final-Holdout未评估。", "",
             "Macro-F1：%，三种子的均值 ± 样本标准差。", "",
             "| 条件 | Baseline | Augmentation | Consistency |", "|---|---:|---:|---:|"]
    for condition in ["clean", *[f"snr{s}" for s in CONDITIONS[1:]]]:
        vals = [next(a for a in aggregates if a["method"] == m and a["condition"] == condition) for m in METHODS]
        lines.append("| " + condition + " | " + " | ".join(
            f"{a['macro_f1_percent_mean']:.2f} ± {a['macro_f1_percent_sample_sd']:.2f}" for a in vals) + " |")
    lines += ["", "健康误报率：%，三种子的均值 ± 样本标准差。", "",
              "| 条件 | Baseline | Augmentation | Consistency |", "|---|---:|---:|---:|"]
    for condition in ["clean", *[f"snr{s}" for s in CONDITIONS[1:]]]:
        vals = [next(a for a in aggregates if a["method"] == m and a["condition"] == condition) for m in METHODS]
        lines.append("| " + condition + " | " + " | ".join(
            f"{a['normal_false_alarm_rate_percent_mean']:.2f} ± {a['normal_false_alarm_rate_percent_sample_sd']:.2f}" for a in vals) + " |")
    lines += ["", "9个最佳权重均按源域验证选定；72份指标经独立实现重算，重载预测逐条一致。",
              "旧Gate0文件哈希未改变。原模型结构、损失、噪声范围及选模规则未修改。", ""]
    (OUT / "SERVER_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    """完整顺序：前置检查→训练→冻结→评估→独立核验；失败留日志，不重试。"""
    os.chdir(CODE)
    assert not OUT.exists(), "不能覆盖已有V1结果"
    OUT.mkdir()
    before = old_fingerprints()
    save(OUT / "RUNTIME.json", {"python": sys.executable, "python_version": sys.version,
         "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
         "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
         "cpu_count": os.cpu_count(), "max_concurrent_jobs": 3, "cpu_threads_per_job": 4,
         "target_holdout_loaded": False, "training_environment_reused": True})
    status("preflight")
    preflight()
    run_pool("train")
    frozen = freeze_weights()
    status("weights_frozen", models=9)
    run_pool("evaluate")
    status("independent_verification")
    verify_and_report(frozen, before)
    status("completed_and_verified", models=9, target_evaluations=72, final_holdout_evaluated=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        status("failed", error=error)
        print(error, file=sys.stderr, flush=True)
        raise
