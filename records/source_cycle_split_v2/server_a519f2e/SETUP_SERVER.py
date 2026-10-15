"""在服务器创建全63文件的独立数据版本；不训练、不触碰旧实验目录。"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT / "code"
PREVIOUS = BASE / "experiments/prototypes_a1_b1/dc8cb36/code"
DATA = BASE / "datasets/source_cycle_split_v2"


def sha(path):
    """只读计算输入及元数据指纹，避免依赖目录绝对路径。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log_name):
    """记录实际命令和退出状态；失败时保留输出，不自动更改数据重试。"""
    result = subprocess.run(command, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=os.environ.copy(), text=True)
    (ROOT / log_name).write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f"命令失败：{command}")
    return result.stdout


def main():
    """复用原59个输入、补入4文件，并复现本地已冻结的周期元数据。"""
    assert not CODE.exists() and not DATA.exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as handle:
        handle.extractall(CODE, filter="data")
    inputs = ROOT / "input63"
    inputs.mkdir()
    protected = {}
    for path in (PREVIOUS / "data/source").glob("*.mat"):
        protected[str(path.resolve())] = sha(path)
        (inputs / path.name).symlink_to(path.resolve())
    for path in (ROOT / "supplemental_source").glob("*.mat"):
        assert not (inputs / path.name).exists()
        (inputs / path.name).symlink_to(path)
    assert len(list(inputs.glob("*.mat"))) == 63
    run([sys.executable, "scripts/source_cycle_split.py", "--source-root", str(inputs),
         "--labels-xlsx", str(ROOT / "source_labels.xlsx"), "--out", str(DATA)], "preparation.log")
    expected = json.loads((ROOT / "CLIENT_METADATA_SHA256.json").read_text())
    for name, digest in expected.items():
        assert sha(DATA / name) == digest, name
    # 新版入口默认目录指向独立数据集；旧冻结数据只保留为显式复现来源。
    (CODE / "outputs").mkdir()
    (CODE / "outputs/source_cycle_split_v2").symlink_to(DATA, target_is_directory=True)
    (CODE / "outputs/gate0_v1").symlink_to(PREVIOUS / "outputs/gate0_v1", target_is_directory=True)
    tests_output = run([sys.executable, "-m", "pytest", "-q", "tests/test_source_cycle_split.py",
         "tests/test_hust_prototypes_a1_b1.py", "tests/test_hust_prototypes_a0_b0.py",
         "tests/test_hust_arch_suite.py", "tests/test_v2_experiment.py", "tests/test_v2_paifilter.py"], "tests.log")
    run(["bash", "-n", "scripts/run_hust_a1_b1_server.sh"], "shell_check.log")
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import source_cycle_split as data
    from v2_experiment import fixed_noisy_validation, fresh_noisy_source
    provenance = data.source_provenance(DATA)
    (raw, train, y, clean), (_, val, vy, val_clean, val_noise) = data.source_views(DATA)
    repeated = fixed_noisy_validation(raw, val, noise_levels=[0, 5, 10, 20, 30])
    assert all(np.array_equal(val_noise[k], repeated[k]) for k in val_noise)
    train_noise = fresh_noisy_source(raw, train, 42, 1)
    assert train_noise.shape == clean.shape == (962, 5, 4096)
    assert val_clean.shape == (221, 5, 4096)
    assert np.isfinite(train_noise).all() and np.isfinite(val_clean).all()
    assert all(sha(path) == digest for path, digest in protected.items())
    result = {
        "status": "passed", "code_commit": (ROOT / "CODE_COMMIT.txt").read_text().strip(),
        "python": sys.executable, "torch_version": str(torch.__version__),
        "existing_uv_environment_reused": True, "dependencies_installed": False,
        "data_root": str(DATA), "code_root": str(CODE), "source_split": provenance,
        "local_server_metadata_exact_match": len(expected), "unit_tests_passed": int(re.search(r"(\d+) passed", tests_output).group(1)),
        "original_source_inputs_sha_unchanged": len(protected),
        "source_train_shape": list(clean.shape), "source_val_shape": list(val_clean.shape),
        "class_counts_train": np.bincount(y, minlength=6).tolist(),
        "class_counts_val": np.bincount(vy, minlength=6).tolist(),
        "fixed_validation_noise_repeatable": True,
        "validation_cycle_sha256": {str(k): hashlib.sha256(x.tobytes()).hexdigest() for k, x in val_noise.items()},
        "first_training_noise_sha256_seed42": hashlib.sha256(train_noise.tobytes()).hexdigest(),
        "model_training_started": False, "target_waveforms_read": False,
    }
    (ROOT / "SERVER_VERIFICATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
