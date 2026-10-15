"""部署A1/B1提交快照；复用已核验输入与uv环境，不改模型或数据划分。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                  OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT / "code"
PREVIOUS = BASE / "experiments/prototypes_a0_b0/c0b30c9"
STRONG = BASE / "datasets/strong_snr_v1_20261012"


def read(path):
    """只读原JSON；历史配置、结果及输入保持原字节。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """流式计算原文件SHA，避免把权重或原始数据加入发布包。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    """只创建本轮新记录；已有运行和失败证据不能覆盖。"""
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    """核对共享读取器、噪声、CUDA梯度和原测试后才开放正式训练。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    (CODE / "data/target_dev").mkdir(parents=True)
    (CODE / "outputs").mkdir()
    for link, target in (
        (CODE / "data/source", PREVIOUS / "code/data/source"),
        (CODE / "data/target_dev/clean", PREVIOUS / "code/data/target_dev/clean"),
        (CODE / "outputs/gate0_v1", PREVIOUS / "code/outputs/gate0_v1"),
    ):
        link.symlink_to(target, target_is_directory=True)
    (ROOT / "extreme_dev").symlink_to(PREVIOUS / "extreme_dev", target_is_directory=True)
    previous = read(PREVIOUS / "DEPLOYMENT_MANIFEST.json")
    unchanged = ("scripts/prepare.py", "scripts/v2_experiment.py",
                 "scripts/v1_noise_consistency.py", "scripts/hust_arch_suite_experiment.py",
                 "scripts/v2_paifilter.py", "scripts/classical_baseline_models.py",
                 "configs/splits_gate0_v1.json", "configs/protocol_gate0_v1.json",
                 "scripts/hust_a0_wavelet_attention.py", "scripts/hust_b0_harmonic_recovery.py")
    for name in unchanged:
        assert sha(CODE / name) == previous["code_sha256"][name], name
    protected = read(PREVIOUS / "PROTECTED_FILES_SHA256.json")
    for folder in (PREVIOUS / "results", PREVIOUS / "code/outputs/gate0_v1"):
        for path in folder.rglob("*"):
            if path.is_file():
                protected[str(path)] = sha(path)
    assert all(sha(path) == expected for path, expected in protected.items())
    extreme = read(PREVIOUS / "EXTREME_DEV_INPUT_INDEX.json")
    assert all(sha(row["original_file"]) == row["sha256"] for row in extreme["files"])
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import hust_a1_b1_experiment as exp
    assert torch.cuda.is_available()
    config = exp.get_config()
    (tr, tm, y, clean), (vr, vm, vy, val_clean, val_noise) = exp.source_views(STRONG)
    assert (len(tr), len(vr), len(y), len(vy)) == (34, 15, 738, 302)
    assert {row["record_id"] for row in tm}.isdisjoint(row["record_id"] for row in vm)
    regenerated = exp.data.fixed_noisy_validation(vr, vm, noise_levels=exp.VAL_NOISE)
    assert all(np.array_equal(val_noise[snr], regenerated[snr]) for snr in exp.VAL_NOISE)
    prior_noise = read(PREVIOUS / "PREFLIGHT.json")["first_epoch_noise_sha256"]
    noise_hashes, initial, counts, gradients = {}, {}, {}, {}
    for seed in exp.SEEDS:
        noisy = exp.data.fresh_noisy_source(tr, tm, seed, 1, noise_levels=exp.TRAIN_NOISE)
        noise_hashes[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noise_hashes[str(seed)] == prior_noise[str(seed)]
        for method in exp.METHODS:
            exp.fix_randomness(seed)
            model = exp.build_model(method).cuda()
            initial[f"{method}/seed{seed}"] = exp.digest_parameters(model)
            counts[method] = exp.count_parameters(model)
            cx, nx = torch.from_numpy(clean[:2]).cuda(), torch.from_numpy(noisy[:2]).cuda()
            labels = torch.from_numpy(y[:2]).cuda()
            loss = exp.b1_paired_loss(model, cx, nx, labels)[0] if method == "b1" else exp.a1_paired_loss(model, cx, nx, labels)
            loss.backward()
            assert torch.isfinite(loss).item()
            assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            gradients[f"{method}/seed{seed}"] = {"loss": float(loss.detach().cpu()), "all_gradients_finite": True}
            # B1恢复预训练也必须在严格确定性CUDA下单独通过反传。
            if method == "b1":
                model.zero_grad(set_to_none=True)
                _, spectrum, variance, _ = model.restorer(nx, return_details=True)
                recovery, _ = exp.b1_recovery_objective(nx, cx, spectrum, variance)
                recovery.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.restorer.parameters())
            del model
    commands = (
        [sys.executable, "-m", "pytest", "-q", "tests/test_hust_prototypes_a1_b1.py", "tests/test_hust_prototypes_a0_b0.py", "tests/test_hust_arch_suite.py", "tests/test_v2_experiment.py", "tests/test_v2_paifilter.py"],
        [sys.executable, "scripts/hust_a1_b1_experiment.py", "smoke"],
        ["bash", "-n", "scripts/run_hust_a1_b1_server.sh"],
    )
    checks = []
    for index, command in enumerate(commands):
        run = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (ROOT / f"preflight_{index}.log").write_text(run.stdout, encoding="utf-8")
        print(run.stdout, flush=True)
        checks.append({"command": command, "exit_code": run.returncode})
        assert run.returncode == 0, command
    save(ROOT / "PROTECTED_FILES_SHA256.json", protected)
    save(ROOT / "EXTREME_DEV_INPUT_INDEX.json", extreme)
    save(ROOT / "DEPLOYMENT_MANIFEST.json", {
        "code_commit": (ROOT / "CODE_COMMIT.txt").read_text().strip(),
        "model_upstream_commit": "d7b6870", "branch": "research/a1-b1-noise-structure-v1",
        "python": sys.executable, "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "environment_reused": True, "dependencies_installed": False,
        "source_strong_root": str(STRONG), "target_strong_root": str(STRONG / "generated/target"),
        "target_extreme_root": str(ROOT / "extreme_dev"), "config": config,
        "parallel_workers": 3, "threads_per_worker": 4, "method_order": ["b1", "a1"],
        "data_split_changed": False, "input_grid_points": 4096,
        "source_test_accessed_for_training_or_validation": False,
        "code_archive_sha256": sha(ROOT / "code.tar"),
        "code_sha256": {p.relative_to(CODE).as_posix(): sha(p) for name in ("scripts", "configs", "tests", "docs")
                        for p in (CODE / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts},
    })
    save(ROOT / "PREFLIGHT.json", {
        "status": "passed", "source_train_records": 34, "source_train_cycles": 738,
        "source_val_records": 15, "source_val_cycles": 302,
        "unchanged_shared_code": list(unchanged), "first_epoch_noise_sha256": noise_hashes,
        "initial_parameters_sha256": initial, "parameter_counts": counts,
        "cuda_backward_checks": gradients, "test_commands": checks,
        "validation_noise_matches_original_rng": True, "old_files_protected": len(protected),
        "final_holdout_accessed": False, "regenerated_mat_files": 0,
    })
    print("A1_B1_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
