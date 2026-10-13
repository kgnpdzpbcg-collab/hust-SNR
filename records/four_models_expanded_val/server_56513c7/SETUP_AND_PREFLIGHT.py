"""四模型扩大验证部署：保持旧文件、训练噪声、CNN初值与模型实现。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT / "code"
V2 = BASE / "experiments/v2_paifilter/4df46a37a6c2"
V4 = BASE / "experiments/v4_hpasf/b4801df"
V5 = BASE / "experiments/v5_nahsf/ae2716b"
STRESS = BASE / "experiments/v4_strong_snr/84f6757"
STRONG = BASE / "datasets/strong_snr_v1_20261012"


def sha(path):
    """记录实际文件字节；全程不改写共享原数据和旧实验结果。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """只保存本轮部署/验证记录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    """测试、固定0/5 dB验证MAT及真实GPU批次全部通过才开始正式实验。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    for p in (CODE / "data/target_dev", CODE / "outputs"):
        p.mkdir(parents=True)
    (CODE / "data/source").symlink_to(V4 / "code/data/source", target_is_directory=True)
    (CODE / "data/target_dev/clean").symlink_to(V4 / "code/data/target_dev/clean", target_is_directory=True)
    (CODE / "outputs/gate0_v1").symlink_to(V4 / "code/outputs/gate0_v1", target_is_directory=True)
    (ROOT / "target_snr").symlink_to(V4 / "target_snr", target_is_directory=True)
    for name, directory in (("v4_hpasf/server_b4801df", V4), ("v4_strong_snr/server_84f6757", STRESS)):
        p = CODE / "records" / name
        p.mkdir(parents=True)
        (p / "results").symlink_to(directory / "results", target_is_directory=True)
    previous = json.loads((V5 / "DEPLOYMENT_MANIFEST.json").read_text())
    for name in ("scripts/v2_paifilter.py", "scripts/v4_hpasf.py", "scripts/v5_nahsf.py",
                 "scripts/v5_experiment.py", "scripts/prepare.py", "scripts/v1_noise_consistency.py", "configs/splits_gate0_v1.json"):
        assert sha(CODE / name) == previous["code_sha256"][name], name
    protected = {str(p): sha(p) for directory in (V2 / "results", V4 / "results", V5 / "results", STRESS / "results",
                   BASE / "experiments/v4_strong_retrain/ddb0638/results") for p in directory.rglob("*") if p.is_file()}
    for name, expected in json.loads((V4 / "ORIGINAL_FILES_SHA256.json").read_text()).items():
        assert sha(BASE / name) == expected
        protected[str(BASE / name)] = expected
    generated = json.loads((STRONG / "generated/GENERATION_MANIFEST.json").read_text())
    assert generated["status"] == "passed" and json.loads((STRONG / "VERIFICATION.json").read_text())["status"] == "passed"
    for row in generated["files"]:
        if (row["domain"] == "simulation" and row["partition"] == "val") or (row["domain"] == "experimental" and row["partition"] == "dev"):
            p = STRONG / "generated" / row["file"]
            assert sha(p) == row["file_sha256"]
            protected[str(p)] = row["file_sha256"]
    code_sha = {p.relative_to(CODE).as_posix(): sha(p) for p in CODE.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".md")}
    reference_roots = {"identity": str(V2), "lowpass3": str(V4), "lowpass4": str(V4), "nahsf": str(V5)}
    write(ROOT / "ORIGINAL_FILES_SHA256.json", protected)
    write(ROOT / "DEPLOYMENT_MANIFEST.json", {"code_commit": "56513c7b37ec990ac1a3aa060ed9b453e8dd7e5d",
          "code_sha256": code_sha, "archive_sha256": sha(ROOT / "code.tar"), "reference_roots": reference_roots,
          "strong_data_root": str(STRONG), "original_pressure_test": str(STRESS), "environment_reused": True,
          "parallel_workers": 4, "protocol": json.loads((CODE / "configs/four_models_expanded_val_v1.json").read_text())})
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_four_models_expanded_val.py", "tests/test_v5_nahsf.py",
               "tests/test_v5_experiment.py", "tests/test_v2_paifilter.py", "tests/test_v2_experiment.py"]
    completed = subprocess.run(command, cwd=CODE, env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout, flush=True)
    assert completed.returncode == 0
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import v2_experiment as data
    from four_models_expanded_val import make_model, METHODS
    from v1_noise_consistency import fix_randomness
    from v2_paifilter import paired_cross_entropy
    raw, meta, y = data.source_data("train")
    vr, vm, _ = data.source_data("val")
    assert (len(raw), len(vr), len(meta), len(vm)) == (34, 15, 738, 302)
    fixed = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5), materialized_root=STRONG / "generated/source")
    regenerated = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5))
    assert all(np.array_equal(fixed[k], regenerated[k]) for k in (0, 5))
    prior = json.loads((V4 / "PREFLIGHT.json").read_text())
    initial, noises, gpu = {}, {}, {}
    clean = data.raw_cycles(raw, meta)
    for seed in (42, 43, 44):
        noisy = data.fresh_noisy_source(raw, meta, seed, 1)
        noises[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noises[str(seed)] == prior["source_initial_noise_sha256"][str(seed)]
        initial[str(seed)] = {}
        for method in METHODS:
            fix_randomness(seed)
            model = make_model(method).cuda()
            initial[str(seed)][method] = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
            old_run = json.loads((Path(reference_roots[method]) / f"results/models/{method}/seed{seed}/run.json").read_text())
            assert initial[str(seed)][method] == old_run["initial_cnn_sha256"]
            if seed == 42:
                optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
                loss = paired_cross_entropy(model, torch.from_numpy(clean[:32]).cuda(), torch.from_numpy(noisy[:32]).cuda(), torch.from_numpy(y[:32]).cuda())
                loss.backward()
                assert torch.isfinite(loss)
                gpu[method] = {"real_cuda_loss": float(loss.detach().cpu())}
                if method == "nahsf":
                    gradient = model.shaping.gate[-1].bias.grad
                    assert gradient is not None and torch.isfinite(gradient).all() and float(gradient.abs().max()) > 0
                    gpu[method]["gate_bias_grad_max"] = float(gradient.abs().max())
                optimizer.step()
            del model
    write(ROOT / "PREFLIGHT.json", {"status": "passed", "unit_test_command": command, "unit_test_output": completed.stdout,
          "initial_cnn_sha256": initial, "first_epoch_noise_sha256": noises, "original_training_noise_unchanged": True,
          "fixed_val0_5_matches_original_rng": True, "real_cuda_checks": gpu,
          "source_train_records": 34, "source_val_records": 15, "source_train_cycles": 738, "source_val_cycles": 302,
          "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__)})
    print("FOUR_MODEL_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
