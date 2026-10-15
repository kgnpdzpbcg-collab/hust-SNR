"""七模型六种子服务器预检；只部署新快照，复用输入和既有uv环境。"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT / "code"
SOURCE = BASE / "datasets/source_cycle_split_v2"
PREVIOUS = BASE / "experiments/prototypes_a1_b1/dc8cb36"
AUTHOR = BASE / "experiments/four_models_expanded_val/56513c7/target_snr"
STRONG = BASE / "datasets/strong_snr_v1_20261012/generated/target"


def read(path):
    """读取冻结状态，不能覆盖已有实验。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """逐块读取原始输入、旧权重和代码，保护实际文件字节。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    """只创建本次独立目录的元数据，不覆盖失败记录。"""
    path = ROOT / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    """验证原模型未改、新划分、六种子噪声和实际GPU梯度后才允许训练。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as handle:
        handle.extractall(CODE, filter="data")
    (CODE / "data/target_dev").mkdir(parents=True)
    (CODE / "outputs").mkdir()
    (CODE / "data/target_dev/clean").symlink_to(PREVIOUS / "code/data/target_dev/clean", target_is_directory=True)
    (CODE / "outputs/gate0_v1").symlink_to(PREVIOUS / "code/outputs/gate0_v1", target_is_directory=True)
    (CODE / "outputs/source_cycle_split_v2").symlink_to(SOURCE, target_is_directory=True)
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import source_cycle_benchmark as exp
    from prepare import OUT
    assert torch.cuda.is_available()
    config = exp.configuration()
    views, provenance = exp.checked_source(SOURCE, config)
    (raw, meta, y, clean), (_, vm, vy, vc, vn) = views
    assert len(raw) == 63 and (len(y), len(vy)) == (962, 221)
    assert sha(SOURCE / "SOURCE_SPLIT.json") == config["source_split_manifest_sha256"]
    protected = {str(p): sha(p) for p in SOURCE.rglob("*") if p.is_file()}
    for folder in (PREVIOUS / "results", BASE / "experiments/prototypes_a0_b0/c0b30c9/results"):
        for p in folder.rglob("*"):
            if p.is_file():
                protected[str(p)] = sha(p)
    for method in ("resnet_lp3", "resnet_lp4"):
        for p in (BASE / "experiments/architecture_suite/5c75bea/results/models" / method).rglob("*"):
            if p.is_file():
                protected[str(p)] = sha(p)
    frozen = read(OUT / "target_dev_clean_windows.json")
    ids = sorted({r["record_id"] for r in frozen})
    assert len(ids) == 88 and len(frozen) == 2474
    protected[str(OUT / "target_dev_clean_windows.json")] = sha(OUT / "target_dev_clean_windows.json")
    # 只为冻结88个编号建立输入索引，不扫描其余347个最终保留记录。
    target_index = []
    for snr in exp.CONDITIONS:
        for rid in ids:
            if snr is None:
                p = CODE / f"data/target_dev/clean/{rid}.mat"
            else:
                base = STRONG if snr in (0, 5) else PREVIOUS / "extreme_dev" if snr in (-5, -10) else AUTHOR
                candidates = [base / f"SNR_{snr}/{rid}.mat", base / f"snr{snr}/{rid}.mat", base / f"{snr}/{rid}.mat"]
                p = next(candidate for candidate in candidates if candidate.is_file())
            protected[str(p)] = sha(p)
            target_index.append({"record_id": rid, "condition": "clean" if snr is None else f"snr{snr}", "path": str(p), "sha256": sha(p)})
    initial, counts, noise_hashes, gradients = {}, {}, {}, {}
    selected = np.asarray([np.flatnonzero(y == label)[0] for label in range(6)])
    for seed in exp.SEEDS:
        noisy = exp.data.fresh_noisy_source(raw, meta, seed, 1, noise_levels=config["train_noise_db"])
        noise_hashes[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        for method in exp.METHODS:
            exp.fix_randomness(seed)
            model = exp.build_model(method).cuda()
            initial[f"{method}/seed{seed}"] = exp.digest_parameters(model)
            counts[method] = sum(p.numel() for p in model.parameters())
            loss, _ = exp.objective(method, model, torch.from_numpy(clean[selected]).cuda(),
                         torch.from_numpy(noisy[selected]).cuda(), torch.from_numpy(y[selected]).cuda(), config)
            loss.backward()
            assert torch.isfinite(loss).item() and all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            gradients[f"{method}/seed{seed}"] = {"loss": float(loss.detach().cpu()), "all_gradients_finite": True}
            del model
    prior = read(BASE / "experiments/source_cycle_split_v2/a519f2e/SERVER_VERIFICATION.json")
    assert noise_hashes["42"] == prior["first_training_noise_sha256_seed42"]
    tests = ["tests/test_source_cycle_benchmark.py", "tests/test_source_cycle_split.py", "tests/test_hust_prototypes_a1_b1.py",
             "tests/test_hust_prototypes_a0_b0.py", "tests/test_hust_arch_suite.py", "tests/test_v2_experiment.py",
             "tests/test_v2_paifilter.py", "tests/test_plain_cnn1d_control.py", "tests/test_xu2025_full_model.py",
             "tests/test_xu2025_full_experiment.py", "tests/test_classical_baselines.py"]
    check = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=CODE, env=os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (ROOT / "tests.log").write_text(check.stdout, encoding="utf-8")
    print(check.stdout, flush=True)
    assert check.returncode == 0
    save("PROTECTED_FILES_SHA256.json", protected)
    save("TARGET_DEV_INPUT_INDEX.json", target_index)
    save("DEPLOYMENT_MANIFEST.json", {"code_commit": (ROOT / "CODE_COMMIT.txt").read_text().strip(),
         "branch": "research/a1-b1-noise-structure-v1", "source_root": str(SOURCE), "snr_root": str(AUTHOR),
         "strong_root": str(STRONG), "extreme_root": str(PREVIOUS / "extreme_dev"), "config": config,
         "python": sys.executable, "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
         "environment_reused": True, "dependencies_installed": False, "parallel_workers": 6, "cpu_threads_per_worker": 2,
         "code_sha256": {p.relative_to(CODE).as_posix(): sha(p) for name in ("scripts", "configs", "tests", "docs")
                         for p in (CODE / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts}})
    save("PREFLIGHT.json", {"status": "passed", "unit_tests_passed": int(re.search(r"(\d+) passed", check.stdout).group(1)),
         "source_split": provenance, "first_training_noise_sha256": noise_hashes, "initial_parameters_sha256": initial,
         "parameters": counts, "cuda_backward_checks": gradients, "source_classes_checked_per_model": 6,
         "target_files_indexed": len(target_index), "final_holdout_accessed": False})
    print("SEVEN_MODELS_SIX_SEEDS_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
