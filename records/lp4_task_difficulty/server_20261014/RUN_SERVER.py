"""独立任务难度实验调度：源域三进程训练、核验冻结后统一实测评估。

复用既有Python环境；保留完整日志、候选与种子。没有自动重试或改参数。
导出代码与结果元数据，生成特征缓存和新/旧分类器文件均留在服务器。
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
REF = BASE / "experiments/architecture_suite/5c75bea"
APRIME = BASE / "experiments/four_models_expanded_val/56513c7"
MINUS5 = BASE / "experiments/architecture_minus5/20261014"
STRONG = BASE / "datasets/strong_snr_v1_20261012"
DATA_MINUS5 = BASE / "datasets/strong_snr_minus5_v1_20261014"
CODE, OUT = ROOT / "code", ROOT / "results"
ENV = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8",
           PYTHONPATH=str(REF / "code/scripts") + ":" + str(CODE / "scripts"))


def read(path):
    """读取冻结来源，不改原实验文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """流式核验实际字节，包含旧权重、旧结果及原始输入指纹。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write(name, value):
    """新调度记录只能写入本次独立目录。"""
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_audit():
    """在实测访问前独立核验九份源域选择、缩放器统计及54组预测。"""
    import joblib
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    sys.path.insert(0, str(CODE / "scripts"))
    sys.path.insert(0, str(REF / "code/scripts"))
    from lp4_difficulty_models import SmallHarmonicMLP
    checks = []
    weights_before = {}
    for seed in (42, 43, 44):
        source = read(OUT / f"reports/source_seed{seed}.json")
        assert source["status"] == "completed_and_verified" and not source["target_data_accessed"]
        with np.load(OUT / f"cache/seed{seed}/source_features.npz", allow_pickle=False) as saved:
            dataset = np.concatenate((saved["clean"], saved["noise"].reshape(-1, 45)))
            assert saved["noise"].shape == (50, 738, 45)
            val = {name: saved["val_" + name].copy() for name in ("clean", "0", "5", "10", "20", "30")}
            vy = saved["val_labels"].copy()
        weights = np.r_[np.ones(738), np.full(738 * 50, 1 / 50)]
        expected_mean = np.average(dataset.astype(np.float64), axis=0, weights=weights)
        expected_var = np.average((dataset.astype(np.float64) - expected_mean)**2, axis=0, weights=weights)
        for method in ("lp4_logreg", "lp4_rbf_svm", "lp4_mlp"):
            folder = OUT / f"models/{method}/seed{seed}"
            record = read(folder / "run.json")
            rows = record["candidates"] if method != "lp4_mlp" else read(folder / "history.json")
            assert len(rows) == (50 if method == "lp4_mlp" else 3)
            top = max(row["clean_f1"] for row in rows)
            eligible = [row for row in rows if row["clean_f1"] >= top - .01 - 1e-12]
            best = sorted(eligible, key=lambda row: (-row["noise_mean_f1"], -row["clean_f1"], row["epoch"]))[0]
            assert best == record["selected"]
            for row in rows:
                assert abs(row["noise_mean_f1"] - np.mean(list(row["noisy_f1"].values()))) < 1e-12
            if method == "lp4_mlp":
                scaler = joblib.load(folder / "scaler.joblib")
                checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                model = SmallHarmonicMLP().eval()
                model.load_state_dict(checkpoint["state"], strict=True)
                def predict(values):
                    with torch.no_grad():
                        return model(torch.from_numpy(values)).argmax(1).numpy()
            else:
                saved = joblib.load(folder / "best.joblib")
                scaler, estimator = saved["scaler"], saved["estimator"]
                predict = estimator.predict
            np.testing.assert_allclose(scaler.mean_, expected_mean, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(scaler.var_, expected_var, atol=1e-10, rtol=1e-10)
            for name, features in val.items():
                predicted = predict(scaler.transform(features).astype(np.float32))
                score = f1_score(vy, predicted, average="macro", labels=range(6), zero_division=0)
                expected = best["clean_f1"] if name == "clean" else best["noisy_f1"][name]
                assert abs(score - expected) < 1e-12, (method, seed, name)
            checks.append({"method": method, "seed": seed, "selected_original_epoch_or_candidate": best["epoch"],
                           "source_clean_macro_f1_percent": best["clean_f1"] * 100,
                           "source_five_noise_macro_f1_mean_percent": best["noise_mean_f1"] * 100})
            for path in folder.iterdir():
                weights_before[str(path)] = sha(path)
    write("PRE_TARGET_SOURCE_VERIFICATION.json", {"status": "passed", "source_selections_independently_recomputed": 9,
          "weighted_scalers_independently_recomputed": 9, "source_selected_prediction_sets_independently_reproduced": 54,
          "selection_records": checks, "target_data_accessed_by_new_training": False, "new_model_files_sha256": weights_before})


def main():
    """所有阶段独立记录；失败不重试、不更改冻结配置、不覆盖旧数据。"""
    started = time.perf_counter()
    assert not CODE.exists() and not OUT.exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    protected = dict(read(DATA_MINUS5 / "records/PROTECTED_FILES_SHA256.json"))
    for folder in (REF / "results", APRIME / "results", MINUS5 / "results"):
        for path in folder.rglob("*"):
            if path.is_file():
                protected[str(path)] = sha(path)
    for path, digest in protected.items():
        assert sha(path) == digest, path
    write("PROTECTED_OLD_FILES_SHA256.json", protected)
    original = read(REF / "DEPLOYMENT_MANIFEST.json")
    for name, digest in original["code_sha256"].items():
        assert sha(REF / "code" / name) == digest, name
    write("DEPLOYMENT_MANIFEST.json", {"client_date": "2026-10-14", "code_commit": (ROOT / "CODE_COMMIT.txt").read_text(encoding="utf-8-sig").strip(),
          "config": read(CODE / "configs/lp4_task_difficulty_v1.json"), "python": sys.executable,
          "environment_reused": True, "dependencies_installed": False, "parallel_train_workers": 3,
          "original_reference_code_sha_verified": True, "reference_architecture": str(REF), "reference_aprime": str(APRIME),
          "reference_minus5": str(MINUS5), "code_archive_sha256": sha(ROOT / "code.tar")})
    command = [sys.executable, "-m", "pytest", "tests/test_lp4_difficulty.py", "-q"]
    with (ROOT / "unit_tests.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=CODE, env=ENV, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    write("UNIT_TEST_COMMAND.json", {"command": command, "cwd": str(CODE), "exit_code": completed.returncode})
    assert completed.returncode == 0, (ROOT / "unit_tests.log").read_text(encoding="utf-8")
    common = [sys.executable, "-u", str(CODE / "scripts/lp4_difficulty_experiment.py")]
    common_args = ["--reference-code", str(REF / "code"), "--output", str(OUT), "--config", str(CODE / "configs/lp4_task_difficulty_v1.json")]
    processes, commands, streams = [], [], []
    for seed in (42, 43, 44):
        command = common + ["train"] + common_args + ["--seed", str(seed), "--source-strong-root", str(STRONG)]
        stream = (ROOT / f"train_seed{seed}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=REF / "code", env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        processes.append(process)
        streams.append(stream)
        commands.append({"seed": seed, "pid": process.pid, "command": command})
    write("TRAINING_COMMANDS.json", commands)
    print("TRAIN_STARTED: three source-only workers", flush=True)
    while any(process.poll() is None for process in processes):
        state = [{"seed": seed, "pid": process.pid, "exit_code": process.poll()} for seed, process in zip((42, 43, 44), processes)]
        write("RUN_STATUS.json", {"phase": "source_training", "workers": state, "elapsed_seconds": time.perf_counter() - started})
        print("TRAIN_PROGRESS: " + json.dumps(state), flush=True)
        time.sleep(15)
        if time.perf_counter() - started > 1800:
            print("HARD_TIMEOUT: terminating source workers after1800s", flush=True)
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            raise TimeoutError("源训练超过冻结30分钟运行上限")
    for stream in streams:
        stream.close()
    exits = [process.returncode for process in processes]
    if any(exits):
        write("RUN_STATUS.json", {"status": "failed", "stage": "source_training", "exit_codes": exits, "target_evaluation_started": False})
        for seed in (42, 43, 44):
            print((ROOT / f"train_seed{seed}.log").read_text(encoding="utf-8")[-8000:], flush=True)
        raise RuntimeError("源训练失败，未重试且未启动实测评估")
    source_audit()
    print("SOURCE_ONLY_SELECTION_VERIFIED: all9 selected classifiers frozen", flush=True)
    deploy = read(REF / "DEPLOYMENT_MANIFEST.json")
    command = common + ["evaluate"] + common_args + ["--target-snr-root", deploy["target_snr_root"],
              "--target-strong-root", str(STRONG), "--minus5-root", str(DATA_MINUS5),
              "--reference-aprime", str(APRIME), "--reference-results", str(REF / "results"),
              "--reference-minus5-results", str(MINUS5 / "results")]
    tick = time.perf_counter()
    with (ROOT / "evaluation.log").open("w", encoding="utf-8") as stream:
        finished = subprocess.run(command, cwd=REF / "code", env=ENV, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    write("EVALUATION_COMMAND.json", {"command": command, "cwd": str(REF / "code"), "exit_code": finished.returncode,
          "elapsed_seconds": time.perf_counter() - tick})
    assert finished.returncode == 0, (ROOT / "evaluation.log").read_text(encoding="utf-8")
    for path, digest in protected.items():
        assert sha(path) == digest, path
    write("RUN_STATUS.json", {"status": "completed_and_verified", "source_only_new_models": 9, "fixed_reference_models": 6,
          "target_evaluation_sets": 165, "protected_old_files_sha_unchanged": len(protected),
          "source_selection_completed_before_target_evaluation": True, "elapsed_seconds": time.perf_counter() - started})
    files = {str(path.relative_to(ROOT)): path for path in ROOT.iterdir() if path.is_file() and path.suffix in (".json", ".log", ".py")}
    for folder in (CODE, OUT):
        for path in folder.rglob("*"):
            if path.is_file() and "cache" not in path.parts and "__pycache__" not in path.parts and path.suffix in (".json", ".csv", ".py", ".md"):
                files[str(path.relative_to(ROOT))] = path
    assert not any(path.suffix in (".pt", ".joblib", ".npz", ".mat") for path in files.values())
    write("RESULT_EXPORT_MANIFEST.json", {"files": {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())},
          "raw_waveforms_included": False, "model_weights_or_feature_caches_included": False})
    files["RESULT_EXPORT_MANIFEST.json"] = ROOT / "RESULT_EXPORT_MANIFEST.json"
    with tarfile.open(ROOT / "lp4_difficulty_records.tar.gz", "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name)
    print("ALL_COMPLETE: nine new classifiers, six fixed references, 165 Dev results and export verified", flush=True)


if __name__ == "__main__":
    main()
