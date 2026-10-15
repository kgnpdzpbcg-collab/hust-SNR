"""部署A0/B0原分支快照，复用原缓存和环境；不改模型、数据划分或超参数。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT / "code"
ARCH = BASE / "experiments/architecture_suite/5c75bea"
STRONG = BASE / "datasets/strong_snr_v1_20261012"
MINUS5 = BASE / "datasets/strong_snr_minus5_v1_20261014"
MINUS10 = BASE / "datasets/strong_snr_minus10_v1_20261015"


def read(path):
    """读取原始配置及指纹记录，不改写已有元数据。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """对实际文件字节计算SHA，保护旧模型、输入和记录。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    """仅创建新记录；重复部署须另分目录，不能覆盖失败证据。"""
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    """验证代码、输入、源噪声及GPU梯度后才允许正式训练。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    (CODE / "data/target_dev").mkdir(parents=True)
    (CODE / "outputs").mkdir()
    for link, target in ((CODE / "data/source", ARCH / "code/data/source"),
                         (CODE / "data/target_dev/clean", ARCH / "code/data/target_dev/clean"),
                         (CODE / "outputs/gate0_v1", ARCH / "code/outputs/gate0_v1")):
        link.symlink_to(target, target_is_directory=True)
    prior = read(ARCH / "DEPLOYMENT_MANIFEST.json")
    unchanged = ("scripts/prepare.py", "scripts/v2_experiment.py", "scripts/v1_noise_consistency.py",
                 "scripts/hust_arch_suite_experiment.py", "scripts/v2_paifilter.py",
                 "scripts/classical_baseline_models.py", "configs/splits_gate0_v1.json")
    for name in unchanged:
        assert sha(CODE / name) == prior["code_sha256"][name], name
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    from scipy.io import loadmat
    import torch
    import hust_prototypes_experiment as exp
    from prepare import OUT
    assert torch.cuda.is_available()
    config = exp.get_config()
    # 仅映射原88个Dev编号；目录不包含347份最终保留波形或重复别名。
    frozen = read(OUT / "target_dev_clean_windows.json")
    ids = sorted({row["record_id"] for row in frozen})
    assert len(ids) == 88 and len(frozen) == 2474
    assert np.bincount([r["label"] for r in frozen], minlength=6).tolist() == [1106, 307, 297, 298, 237, 229]
    protected = {}
    for folder in (ARCH / "results", BASE / "experiments/architecture_minus5/20261014/results",
                   BASE / "experiments/resnet_lp4_minus10/20261015/results"):
        for path in folder.rglob("*"):
            if path.is_file():
                protected[str(path)] = sha(path)
    for path in (ARCH / "code/outputs/gate0_v1").rglob("*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    index, max_error = [], 0.0
    for snr, data_root in ((-5, MINUS5), (-10, MINUS10)):
        manifest = read(data_root / "generated/GENERATION_MANIFEST.json")
        entries = {r["record_id"]: r for r in manifest["files"] if r["domain"] == "experimental" and r["partition"] == "dev"}
        assert set(entries) == set(ids)
        destination = ROOT / "extreme_dev" / f"SNR_{snr}"
        destination.mkdir(parents=True)
        for rid in ids:
            path = data_root / "generated" / entries[rid]["file"]
            assert sha(path) == entries[rid]["file_sha256"]
            noisy = loadmat(path)["I_noisy"]
            clean_path = CODE / "data/target_dev/clean" / f"{rid}.mat"
            author_path = STRONG / "inputs/target_snr10" / f"{rid}.mat"
            clean, author = loadmat(clean_path)["name"], loadmat(author_path)["I_noisy"]
            # 调用新分支的同一加噪公式，证明复用波形与其生成器逐数值相同。
            from generate_extreme_dev_snr import scale_author_residual, full_record_phase_snr
            assert np.array_equal(noisy[:, 1:6], scale_author_residual(clean[:, 1:6], author[:, 1:6], snr))
            assert np.array_equal(noisy[:, [0, -1]], clean[:, [0, -1]])
            max_error = max(max_error, float(np.max(np.abs(full_record_phase_snr(clean[:, 1:6], noisy[:, 1:6]) - snr))))
            (destination / f"{rid}.mat").symlink_to(path)
            for original in (path, clean_path, author_path):
                protected[str(original)] = sha(original)
            index.append({"snr_db": snr, "record_id": rid, "original_file": str(path), "sha256": sha(path)})
    (tr, tm, y, clean), (vr, vm, vy, val_clean, val) = exp.source_views(STRONG)
    assert (len(tr), len(vr), len(y), len(vy)) == (34, 15, 738, 302)
    assert {r["record_id"] for r in tm}.isdisjoint(r["record_id"] for r in vm)
    for rid in set(tr) | set(vr):
        path = CODE / "data/source" / f"{rid}.mat"
        protected[str(path)] = sha(path)
    for snr in (0, 5):
        for rid in vr:
            path = STRONG / "generated/source" / f"SNR_{snr}/{rid}.mat"
            protected[str(path)] = sha(path)
        for rid in ids:
            path = STRONG / "generated/target" / f"SNR_{snr}/{rid}.mat"
            protected[str(path)] = sha(path)
    original = exp.data.fixed_noisy_validation(vr, vm, noise_levels=exp.VAL_NOISE)
    assert all(np.array_equal(val[snr], original[snr]) for snr in exp.VAL_NOISE)
    prior_noise = read(ARCH / "PREFLIGHT.json")["first_epoch_noise_sha256"]
    noise_hashes, initial, counts, gpu = {}, {}, {}, {}
    for seed in exp.SEEDS:
        noisy = exp.data.fresh_noisy_source(tr, tm, seed, 1, noise_levels=exp.TRAIN_NOISE)
        digest = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert digest == prior_noise[str(seed)]
        noise_hashes[str(seed)] = digest
        for method in exp.METHODS:
            exp.fix_randomness(seed)
            model = exp.build_model(method).cuda()
            initial[f"{method}/seed{seed}"] = exp.digest_parameters(model)
            counts[method] = exp.count_parameters(model)
            cx, nx = torch.from_numpy(clean[:2]).cuda(), torch.from_numpy(noisy[:2]).cuda()
            labels = torch.from_numpy(y[:2]).cuda()
            loss = exp.b0_paired_loss(model, cx, nx, labels)[0] if method == "b0" else exp.a0_paired_loss(model, cx, nx, labels)
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            gpu[f"{method}/seed{seed}"] = {"loss": float(loss.detach().cpu()), "gradients_finite": True}
            del model
    for i, command in enumerate(([sys.executable, "-m", "pytest", "-q", "tests/test_hust_prototypes_a0_b0.py"],
                                  [sys.executable, "scripts/hust_prototypes_experiment.py", "smoke"],
                                  ["bash", "-n", "scripts/run_hust_prototypes_server.sh"])):
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (ROOT / f"preflight_{i}.log").write_text(result.stdout, encoding="utf-8")
        print(result.stdout, flush=True)
        assert result.returncode == 0, command
    save(ROOT / "PROTECTED_FILES_SHA256.json", protected)
    save(ROOT / "EXTREME_DEV_INPUT_INDEX.json", {"files": index, "linked_files": 176,
         "regenerated_mat_files": 0, "matches_branch_generator_exactly": True,
         "max_full_record_phase_snr_error_db": max_error, "final_holdout_accessed": False})
    save(ROOT / "DEPLOYMENT_MANIFEST.json", {"code_commit": (ROOT / "CODE_COMMIT.txt").read_text().strip(),
         "branch": "research/harmonic-recovery-wavelet-a0-b0", "client_date": "2026-10-15",
         "code_archive_sha256": sha(ROOT / "code.tar"), "environment_reused": True,
         "dependencies_installed": False, "python": sys.executable, "torch_version": str(torch.__version__),
         "gpu": torch.cuda.get_device_name(), "source_strong_root": str(STRONG),
         "target_strong_root": str(STRONG / "generated/target"), "target_extreme_root": str(ROOT / "extreme_dev"),
         "config": config, "parallel_workers": 3, "threads_per_worker": 4,
         "method_order": ["b0", "a0"], "algorithm_or_frozen_config_modified": False,
         "code_sha256": {p.relative_to(CODE).as_posix(): sha(p) for folder in ("scripts", "configs", "tests", "docs")
                         for p in (CODE / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts}})
    save(ROOT / "PREFLIGHT.json", {"status": "passed", "unit_tests_passed": 10,
         "shared_data_readers_match_original_code_sha": list(unchanged),
         "source_train_records": 34, "source_train_cycles": 738, "source_val_records": 15,
         "source_val_cycles": 302, "source_val_noise_matches_original_rng": True,
         "first_epoch_noise_sha256": noise_hashes, "source_noise_matches_historical_baseline": True,
         "initial_parameters_sha256": initial, "parameter_counts": counts,
         "gpu_backward_checks": gpu, "extreme_noise_identical_to_branch_generator": True,
         "old_files_protected": len(protected), "final_holdout_accessed": False})
    print("A0_B0_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
