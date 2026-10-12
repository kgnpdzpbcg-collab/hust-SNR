"""部署独立目录并核查输入、旧文件、默认协议和真实CUDA训练批次。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[2] / "experiments/v4_hpasf/b4801df"
PRESSURE = ROOT.parents[2] / "experiments/v4_strong_snr/84f6757"
STRONG = ROOT.parents[2] / "datasets/strong_snr_v1_20261012"
CODE = ROOT / "code"


def sha(path):
    """对部署代码、原数据与旧结果做完整字节核查。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    """本轮元数据仅写新实验目录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    """先通过单元测试和实际输入检查，再允许正式训练。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    for parent in (CODE / "data/target_dev", CODE / "outputs", ROOT / "target_snr"):
        parent.mkdir(parents=True)
    (CODE / "data/source").symlink_to(OLD / "code/data/source", target_is_directory=True)
    (CODE / "data/target_dev/clean").symlink_to(OLD / "code/data/target_dev/clean", target_is_directory=True)
    (CODE / "outputs/gate0_v1").symlink_to(OLD / "code/outputs/gate0_v1", target_is_directory=True)
    for level in (0, 5, 10, 15, 20, 25, 30, 35, 40):
        source = STRONG / f"generated/target/SNR_{level}" if level in (0, 5) else OLD / f"target_snr/SNR_{level}"
        assert source.is_dir()
        (ROOT / f"target_snr/SNR_{level}").symlink_to(source, target_is_directory=True)
    old_deployment = json.loads((OLD / "DEPLOYMENT_MANIFEST.json").read_text())
    for name in ("scripts/v4_hpasf.py", "scripts/v2_paifilter.py", "scripts/v1_noise_consistency.py", "scripts/prepare.py", "configs/splits_gate0_v1.json"):
        assert sha(CODE / name) == old_deployment["code_sha256"][name], name
    assert json.loads((STRONG / "VERIFICATION.json").read_text())["status"] == "passed"
    manifest = json.loads((STRONG / "generated/GENERATION_MANIFEST.json").read_text())
    assert sha(CODE / "configs/splits_gate0_v1.json") == manifest["split_sha256"]
    # 保护旧权重/日志/评估以及实际使用的原MAT、新MAT；不扫描最终保留波形。
    protected = [p for root in (OLD / "results", PRESSURE / "results") for p in root.rglob("*") if p.is_file()]
    protected += [OLD / "code" / p for p in old_deployment["code_sha256"]]
    protected += [p for p in (CODE / "data/source").glob("*.mat")]
    protected += [p for p in (CODE / "data/target_dev/clean").glob("*.mat")]
    protected += [p for level in (10, 15, 20, 25, 30, 35, 40) for p in (ROOT / f"target_snr/SNR_{level}").glob("*.mat")]
    for row in manifest["files"]:
        if (row["domain"] == "simulation" and row["partition"] == "val") or (row["domain"] == "experimental" and row["partition"] == "dev"):
            path = STRONG / "generated" / row["file"]
            assert sha(path) == row["file_sha256"]
            protected.append(path)
    write(ROOT / "ORIGINAL_FILES_SHA256.json", {str(p): sha(p) for p in protected})
    code_sha = {str(p.relative_to(CODE)): sha(p) for p in CODE.rglob("*") if p.is_file() and p.suffix in (".py", ".json")}
    write(ROOT / "DEPLOYMENT_MANIFEST.json", {"code_commit": "ddb0638c72507efaecbc0b99722d1a79e1e536a9",
         "code_sha256": code_sha, "archive_sha256": sha(ROOT / "code.tar"),
         "original_v4": str(OLD), "original_pressure_test": str(PRESSURE), "strong_data_root": str(STRONG),
         "environment_reused": True, "parallel_workers": 4,
         "protocol": json.loads((CODE / "configs/v4_strong_noise_v1.json").read_text())})
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_v4_hpasf.py", "tests/test_v4_experiment.py",
               "tests/test_v2_paifilter.py", "tests/test_v2_experiment.py", "tests/test_v4_strong_noise_experiment.py"]
    result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    assert result.returncode == 0
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import v2_experiment as data
    from v4_strong_noise_experiment import config
    from v4_hpasf import HPASFCNN, METHODS
    from v2_paifilter import paired_cross_entropy
    from v1_noise_consistency import fix_randomness
    protocol = config()
    tr, tm, ty = data.source_data("train")
    vr, vm, vy = data.source_data("val")
    assert (len(tr), len(vr), len(tm), len(vm)) == (34, 15, 738, 302)
    saved_val = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5), materialized_root=STRONG / "generated/source")
    replay_val = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5))
    assert all(np.array_equal(saved_val[k], replay_val[k]) for k in (0, 5))
    clean = data.raw_cycles(tr, tm)
    initial, noises, gradients = {}, {}, {}
    for seed in (42, 43, 44):
        noisy = data.fresh_noisy_source(tr, tm, seed, 1, noise_levels=protocol["training_noise_levels_db"])
        noises[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        initial[str(seed)] = {}
        for method in METHODS:
            fix_randomness(seed)
            model = HPASFCNN(method).cuda()
            fingerprint = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
            original = json.loads((OLD / f"results/models/{method}/seed{seed}/run.json").read_text())
            assert fingerprint == original["initial_cnn_sha256"]
            initial[str(seed)][method] = fingerprint
            if seed == 42:
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                loss = paired_cross_entropy(model, torch.from_numpy(clean[:32]).cuda(), torch.from_numpy(noisy[:32]).cuda(), torch.from_numpy(ty[:32]).cuda())
                assert torch.isfinite(loss)
                loss.backward()
                gradients[method] = {"real_cuda_training_loss": float(loss.detach().cpu())}
                if method in ("static4", "hpasf"):
                    parameter = model.shaping.gate_logit
                    before = parameter.detach().clone()
                    assert parameter.grad is not None and torch.isfinite(parameter.grad) and float(parameter.grad.abs()) > 0
                    gradients[method]["gate_gradient"] = float(parameter.grad.detach().cpu())
                    optimizer.step()
                    assert not torch.equal(before, parameter.detach())
            del model
    write(ROOT / "PREFLIGHT.json", {"status": "passed", "unit_test_command": command, "unit_test_output": result.stdout,
         "source_train_records": 34, "source_val_records": 15, "train_cycles": 738, "val_cycles": 302,
         "materialized_0_5_val_matches_fixed_rng": True, "initial_cnn_sha256": initial,
         "first_epoch_noisy_cycles_sha256": noises, "real_cuda_gradient_checks": gradients,
         "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__)})
    print("STRONG_NOISE_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
