"""服务器生成并独立核验−5 dB数据；不训练、不推理、不改旧实验配置。

输入沿用已核验的59条仿真及440对实测原文件；新MAT留在独立目录。
导出包只包含配置、代码、命令、文件清单和核验记录，不包含波形或权重。
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
from collections import Counter

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
import numpy as np
import scipy
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
OLD = BASE / "datasets/strong_snr_v1_20261012"
ARCH = BASE / "experiments/architecture_suite/5c75bea"
CODE = ROOT / "code"
OUT = ROOT / "generated"
REC = ROOT / "records"


def read(path):
    """读取已保存的元数据，不重写原文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    """完整文件指纹用于验证旧文件未变及导出传输完整性。"""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    """记录客观配置、计数和误差；全部新记录写入独立目录。"""
    (REC / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command, log_name, cwd):
    """保存实际命令、完整输出和用时，失败立即停止后续步骤。"""
    print(f"START: {log_name}", flush=True)
    started = time.perf_counter()
    with (REC / log_name).open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    result = {"command": list(map(str, command)), "cwd": str(cwd),
              "exit_code": completed.returncode, "elapsed_seconds": time.perf_counter() - started}
    if completed.returncode:
        print((REC / log_name).read_text(encoding="utf-8"), flush=True)
        raise RuntimeError(f"步骤失败：{log_name}")
    print(f"COMPLETE: {log_name}, {result['elapsed_seconds']:.3f}s", flush=True)
    return result


def verify(protected):
    """重新读499个新文件，独立复算SNR、原列、噪声公式及固定源Val随机流。"""
    manifest = read(OUT / "GENERATION_MANIFEST.json")
    split = read(OLD / "metadata/splits_gate0_v1.json")
    labels = read(OLD / "metadata/LABELS.json")
    assert manifest["file_count"] == 499 and manifest["source_files"] == 59
    assert manifest["target_files"] == 440
    assert len(list(OUT.rglob("*.mat"))) == 499
    source_part = {rid: part for part, ids in split["source_records"].items() for rid in ids}
    dev, hold = set(split["target_dev_records"]), set(split["target_holdout_records"])
    canonical = {rid: rid for rid in labels["experimental"]}
    for group in split["target_duplicate_groups"]:
        canonical.update({rid: min(group) for rid in group})
    counts, by_domain = Counter(), Counter()
    max_snr_error = 0.0
    max_relative_zero_error = 0.0
    val_checks = 0
    sys.path.insert(0, str(CODE / "scripts"))
    # 只借用旧数据增强函数进行逐数值对照，不构建或调用任何模型。
    from v2_experiment import calibrated_noise
    for row in manifest["files"]:
        rid = row["record_id"]
        source = row["domain"] == "simulation"
        field = "I_We" if source else "I_noisy"
        clean_path = BASE / "data/source" / f"{rid}.mat" if source else OLD / "inputs/target_clean" / f"{rid}.mat"
        clean = loadmat(clean_path)["I_We" if source else "name"]
        generated = OUT / row["file"]
        noisy = loadmat(generated)[field]
        assert sha(generated) == row["file_sha256"]
        assert hashlib.sha256(np.ascontiguousarray(noisy).tobytes()).hexdigest() == row["matrix_sha256"]
        assert row["snr_db"] == -5 and noisy.shape == ((10000, 6) if source else (10031, 7))
        assert noisy.dtype == np.float64 and np.isfinite(noisy).all()
        assert np.array_equal(noisy[:, -1], clean[:, -1])
        cols = slice(0, 5) if source else slice(1, 6)
        measured = 10 * np.log10(np.mean(clean[:, cols] ** 2, axis=0) /
                                 np.mean((noisy[:, cols] - clean[:, cols]) ** 2, axis=0))
        max_snr_error = max(max_snr_error, float(np.max(np.abs(measured + 5))))
        if source:
            part = source_part[rid]
            key = f"HUST-V1:val:-5:{rid}" if part == "val" else f"HUST-STRONG-v1:{part}:-5:{rid}:20261012"
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
            assert seed == row["rng_seed"]
            current = clean[:, :5].copy()
            eps = np.random.default_rng(seed).standard_normal(current.shape)
            eps *= np.sqrt(np.mean(current**2, axis=0) / np.mean(eps**2, axis=0)) / 10**(-5/20)
            assert np.array_equal(noisy[:, :5], current + eps)
            if part == "val":
                # 原函数使用读取器copy后的行连续电流，确保MAT内存排列不造成数值差异。
                raw = {rid: (np.arange(len(current)), current, clean[:, -1].copy())}
                expected = calibrated_noise(raw, {rid: -5}, lambda _: np.random.default_rng(seed))
                assert np.array_equal(noisy[:, :5], expected[rid][1])
                val_checks += 1
        else:
            part = "dev" if rid in dev else "final_holdout" if rid in hold else "duplicate_alias"
            assert row["canonical_record_id"] == canonical[rid]
            author = loadmat(OLD / "inputs/target_snr10" / f"{rid}.mat")["I_noisy"]
            expected = clean[:, 1:6] + (author[:, 1:6] - clean[:, 1:6]) * 10**0.75
            assert np.array_equal(noisy[:, 1:6], expected)
            assert np.array_equal(noisy[:, [0, -1]], clean[:, [0, -1]])
            # 与已生成0 dB残差的1.778倍作独立对照；允许浮点减法末位舍入。
            zero = loadmat(OLD / "generated/target/SNR_0" / f"{rid}.mat")["I_noisy"]
            residual = noisy[:, 1:6] - clean[:, 1:6]
            scaled_zero = (zero[:, 1:6] - clean[:, 1:6]) * 10**0.25
            error = float(np.max(np.abs(residual - scaled_zero)))
            max_relative_zero_error = max(max_relative_zero_error, error)
            assert np.allclose(residual, scaled_zero, atol=1e-12, rtol=1e-13)
        assert row["partition"] == part
        counts[f"{row['domain']}/{part}"] += 1
        by_domain[row["domain"]] += 1
    assert counts == {"simulation/train": 34, "simulation/val": 15, "simulation/source_test": 10,
                      "experimental/dev": 88, "experimental/final_holdout": 347, "experimental/duplicate_alias": 5}
    assert val_checks == 15 and max_snr_error < 1e-6
    for name, expected in protected.items():
        assert sha(Path(name)) == expected, name
    result = {"status": "passed", "generated_mat_files_verified": 499,
              "partition_file_counts": dict(sorted(counts.items())),
              "max_phase_snr_error_db": max_snr_error,
              "all_source_matrices_exactly_reproduced_from_fixed_seeds": True,
              "source_val_records_exactly_match_original_calibrated_noise": val_checks,
              "all_target_currents_exactly_match_scaled_author10_formula": True,
              "max_target_residual_difference_vs_zero_times_10_power_025": max_relative_zero_error,
              "time_and_angular_speed_columns_unchanged": True,
              "raw_row_counts_and_duplicate_timestamps_preserved": True,
              "original_input_mat_files_unchanged": 939, "old_zero_five_mat_files_unchanged": 998,
              "protected_old_files_unchanged": len(protected),
              "protected_model_weights_unchanged": sum(Path(p).suffix in (".pt", ".pth") for p in protected),
              "models_trained": 0, "model_inference_evaluations": 0,
              "training_validation_protocol_modified": False,
              "verification_script_sha256": sha(Path(__file__))}
    write("INDEPENDENT_VERIFICATION.json", result)
    with (REC / "GENERATED_FILE_INDEX.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "domain", "record_id", "partition", "snr_db", "bytes", "file_sha256", "matrix_sha256"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in manifest["files"])
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    """只创建新目录；先保护旧数据/结果，再生成、核验、导出元数据。"""
    assert not OUT.exists() and not CODE.exists() and not REC.exists()
    REC.mkdir()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    inputs = read(OLD / "metadata/INPUT_MANIFEST.json")
    prior = read(OLD / "generated/GENERATION_MANIFEST.json")
    protected = dict(read(ARCH / "PROTECTED_FILES_SHA256.json"))
    for path in (ARCH / "results").rglob("*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    for name, digest in inputs["source_clean_sha256"].items():
        protected[str(BASE / "data/source" / name)] = digest
    for name, entry in inputs["target_files"].items():
        protected[str(OLD / "inputs" / name)] = entry["sha256"]
    for row in prior["files"]:
        protected[str(OLD / "generated" / row["file"])] = row["file_sha256"]
    for path in list((OLD / "metadata").glob("*.json")) + [OLD / "generated/GENERATION_MANIFEST.json"]:
        protected[str(path)] = sha(path)
    for name, digest in protected.items():
        assert sha(Path(name)) == digest, name
    write("PROTECTED_FILES_SHA256.json", protected)
    write("DEPLOYMENT_MANIFEST.json", {"client_date": "2026-10-14", "code_commit": (ROOT / "CODE_COMMIT.txt").read_text(encoding="utf-8-sig").strip(),
          "code_archive_sha256": sha(ROOT / "code.tar"), "python_executable": sys.executable,
          "python_version": platform.python_version(), "numpy_version": np.__version__, "scipy_version": scipy.__version__,
          "new_root": str(ROOT), "original_inputs_root": str(OLD), "source_root": str(BASE / "data/source"),
          "old_environment_reused": True, "dependencies_installed": False})
    write("UNIT_TEST_COMMAND.json", run([sys.executable, "-m", "pytest", "tests/test_generate_strong_snr.py", "-q"], "unit_tests.log", CODE))
    command = [sys.executable, "-u", str(CODE / "scripts/generate_strong_snr.py"),
               "--source-root", str(BASE / "data/source"),
               "--target-clean-root", str(OLD / "inputs/target_clean"),
               "--target-snr10-root", str(OLD / "inputs/target_snr10"),
               "--labels", str(OLD / "metadata/LABELS.json"),
               "--split", str(OLD / "metadata/splits_gate0_v1.json"),
               "--config", str(CODE / "configs/strong_snr_minus5_v1.json"),
               "--input-manifest", str(OLD / "metadata/INPUT_MANIFEST.json"), "--output", str(OUT)]
    write("GENERATION_COMMAND.json", run(command, "generation.log", CODE))
    started = time.perf_counter()
    verify(protected)
    write("VERIFICATION_COMMAND.json", {"command": [sys.executable, "-u", str(Path(__file__))],
          "independent_verification_elapsed_seconds": time.perf_counter() - started})
    # 所有导出对象按原字节复制；大型生成MAT、源波形和模型权重不打包。
    for name, path in {"GENERATION_MANIFEST.json": OUT / "GENERATION_MANIFEST.json",
                       "strong_snr_minus5_v1.json": CODE / "configs/strong_snr_minus5_v1.json",
                       "RUN_MINUS5.py": Path(__file__),
                       "generate_strong_snr.py": CODE / "scripts/generate_strong_snr.py",
                       "test_generate_strong_snr.py": CODE / "tests/test_generate_strong_snr.py"}.items():
        (REC / name).write_bytes(path.read_bytes())
    export = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(REC.iterdir()) if path.is_file()}
    write("RECORD_EXPORT_MANIFEST.json", {"files": export, "raw_mat_included": False, "model_weights_included": False})
    with tarfile.open(ROOT / "minus5_records.tar.gz", "w:gz") as archive:
        for path in sorted(REC.iterdir()):
            archive.add(path, arcname=path.name)
    print("ALL_COMPLETE: 499 MAT generated and verified; metadata archive ready", flush=True)


if __name__ == "__main__":
    main()
