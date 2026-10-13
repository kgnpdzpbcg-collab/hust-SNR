"""固定半增益部署：复用旧环境和数据，执行公平性与真实CUDA批次核验。"""
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
REFERENCE = BASE / "experiments/four_models_expanded_val/56513c7"
V4 = BASE / "experiments/v4_hpasf/b4801df"
STRONG = BASE / "datasets/strong_snr_v1_20261012"


def sha(path):
    """流式SHA记录数据/权重字节，不重新生成或改写共享文件。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """仅保存本轮独立目录内的客观部署与检查记录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    """全部预检通过后才允许正式训练；之前的V5结果不参与选模。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    (CODE / "data/target_dev").mkdir(parents=True)
    (CODE / "outputs").mkdir()
    (CODE / "data/source").symlink_to(REFERENCE / "code/data/source", target_is_directory=True)
    (CODE / "data/target_dev/clean").symlink_to(REFERENCE / "code/data/target_dev/clean", target_is_directory=True)
    (CODE / "outputs/gate0_v1").symlink_to(REFERENCE / "code/outputs/gate0_v1", target_is_directory=True)
    (ROOT / "target_snr").symlink_to(REFERENCE / "target_snr", target_is_directory=True)
    # 历史结果测试依赖两组公开JSON；此处只读链接，绝不改写历史文件。
    for name, directory in (("v4_hpasf/server_b4801df", V4),
                            ("v4_strong_snr/server_84f6757", BASE / "experiments/v4_strong_snr/84f6757")):
        p = CODE / "records" / name
        p.mkdir(parents=True)
        (p / "results").symlink_to(directory / "results", target_is_directory=True)
    previous = json.loads((REFERENCE / "DEPLOYMENT_MANIFEST.json").read_text())
    assert previous["code_commit"] == "56513c7b37ec990ac1a3aa060ed9b453e8dd7e5d"
    for name in ("scripts/v2_experiment.py", "scripts/v2_paifilter.py", "scripts/v4_hpasf.py",
                 "scripts/v5_nahsf.py", "scripts/v5_experiment.py", "scripts/prepare.py",
                 "scripts/v1_noise_consistency.py", "configs/splits_gate0_v1.json",
                 "configs/four_models_expanded_val_v1.json"):
        assert sha(CODE / name) == previous["code_sha256"][name], name
    assert json.loads((REFERENCE / "results/VERIFICATION.json").read_text())["status"] == "passed"
    protected = dict(json.loads((REFERENCE / "ORIGINAL_FILES_SHA256.json").read_text()))
    # 保护最新四模型的全部权重和数值结果；包含V5即将复用的三份best.pt。
    protected.update({str(p): sha(p) for p in (REFERENCE / "results").rglob("*") if p.is_file()})
    assert all(sha(name) == expected for name, expected in protected.items())
    write(ROOT / "ORIGINAL_FILES_SHA256.json", protected)
    write(ROOT / "DEPLOYMENT_MANIFEST.json", {
        "code_commit": "3c3803704d332e81448cfb1afcef1fbd9fe55a28",
        "code_sha256": {p.relative_to(CODE).as_posix(): sha(p) for p in CODE.rglob("*")
                        if p.is_file() and p.suffix in (".py", ".json", ".md")},
        "archive_sha256": sha(ROOT / "code.tar"), "reference_root": str(REFERENCE),
        "reference_code_commit": previous["code_commit"], "strong_data_root": str(STRONG),
        "environment_reused": True, "python_executable": sys.executable, "parallel_workers": 3,
        "protocol": json.loads((CODE / "configs/fixed_half_vs_v5_v1.json").read_text())})
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_fixed_half_fusion.py",
               "tests/test_four_models_expanded_val.py", "tests/test_v5_nahsf.py", "tests/test_v5_experiment.py",
               "tests/test_v2_paifilter.py", "tests/test_v2_experiment.py"]
    done = subprocess.run(command, cwd=CODE, env=os.environ.copy(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(done.stdout, flush=True)
    assert done.returncode == 0
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import v2_experiment as data
    import fixed_half_experiment as experiment
    from v5_nahsf import NAHSFCNN
    from v1_noise_consistency import fix_randomness
    from v2_paifilter import paired_cross_entropy
    tr, tm, y = data.source_data("train")
    vr, vm, _ = data.source_data("val")
    assert (len(tr), len(vr), len(tm), len(vm)) == (34, 15, 738, 302)
    fixed = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5), materialized_root=STRONG / "generated/source")
    regenerated = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5))
    assert all(np.array_equal(fixed[k], regenerated[k]) for k in (0, 5))
    prior = json.loads((REFERENCE / "PREFLIGHT.json").read_text())
    initial, noises, checks = {}, {}, {}
    clean = data.raw_cycles(tr, tm)
    for seed in (42, 43, 44):
        noisy = data.fresh_noisy_source(tr, tm, seed, 1)
        noises[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noises[str(seed)] == prior["first_epoch_noise_sha256"][str(seed)]
        fix_randomness(seed)
        model = experiment.make_model("fixed_half").cuda().eval()
        initial[str(seed)] = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
        assert initial[str(seed)] == prior["initial_cnn_sha256"][str(seed)]["nahsf"]
        fix_randomness(seed)
        v5 = NAHSFCNN().cuda().eval()
        inputs = torch.from_numpy(clean[:32]).cuda()
        with torch.no_grad():
            assert torch.equal(model.shaping(inputs), v5.shaping(inputs))
            assert torch.equal(model(inputs), v5(inputs))
        del v5
        model.train()
        loss = paired_cross_entropy(model, inputs, torch.from_numpy(noisy[:32]).cuda(), torch.from_numpy(y[:32]).cuda())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert not list(model.shaping.parameters())
        checks[str(seed)] = {"real_cuda_loss": float(loss.detach().cpu()), "initial_logits_equal_v5": True,
                              "finite_cnn_gradient": True, "spectral_parameters": 0}
        torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.).step()
        del model
    write(ROOT / "PREFLIGHT.json", {"status": "passed", "unit_test_command": command, "unit_test_output": done.stdout,
          "initial_cnn_sha256": initial, "first_epoch_noise_sha256": noises,
          "fixed_val0_5_matches_original_rng": True, "real_cuda_checks": checks,
          "source_train_records": 34, "source_val_records": 15, "source_train_cycles": 738, "source_val_cycles": 302,
          "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__)})
    print("FIXED_HALF_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
