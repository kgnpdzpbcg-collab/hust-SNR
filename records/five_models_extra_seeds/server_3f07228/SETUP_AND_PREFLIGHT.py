"""部署五模型新增种子：所有原文件和训练循环保持不变，执行GPU公平性核验。"""
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
FOUR = BASE / "experiments/four_models_expanded_val/56513c7"
FIXED = BASE / "experiments/fixed_half_fusion/3c38037"
V4 = BASE / "experiments/v4_hpasf/b4801df"
STRONG = BASE / "datasets/strong_snr_v1_20261012"


def sha(path):
    """流式读取原始字节，保护旧数据、权重和结果不被本轮改写。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    """部署元数据仅写入本轮独立目录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    """旧实验核验、54项测试、15个GPU模型检查通过才允许正式训练。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    (CODE / "data/target_dev").mkdir(parents=True)
    (CODE / "outputs").mkdir()
    for link, target in ((CODE / "data/source", FOUR / "code/data/source"),
                         (CODE / "data/target_dev/clean", FOUR / "code/data/target_dev/clean"),
                         (CODE / "outputs/gate0_v1", FOUR / "code/outputs/gate0_v1"),
                         (ROOT / "target_snr", FOUR / "target_snr")):
        link.symlink_to(target, target_is_directory=True)
    for name, directory in (("v4_hpasf/server_b4801df", V4),
                            ("v4_strong_snr/server_84f6757", BASE / "experiments/v4_strong_snr/84f6757")):
        p = CODE / "records" / name
        p.mkdir(parents=True)
        (p / "results").symlink_to(directory / "results", target_is_directory=True)
    previous = json.loads((FIXED / "DEPLOYMENT_MANIFEST.json").read_text())
    assert previous["code_commit"] == "3c3803704d332e81448cfb1afcef1fbd9fe55a28"
    # 训练、数据读取、模型和旧配置逐文件SHA核对；只新增本轮入口/配置。
    frozen_names = ["scripts/" + name for name in ("four_models_expanded_val.py", "fixed_half_experiment.py",
        "fixed_half_fusion.py", "v2_experiment.py", "v2_paifilter.py", "v4_hpasf.py", "v5_nahsf.py",
        "v5_experiment.py", "prepare.py", "v1_noise_consistency.py")]
    frozen_names += ["configs/" + name for name in ("splits_gate0_v1.json", "four_models_expanded_val_v1.json", "fixed_half_vs_v5_v1.json")]
    for name in frozen_names:
        assert sha(CODE / name) == previous["code_sha256"][name], name
    assert all(json.loads((p / "results/VERIFICATION.json").read_text())["status"] == "passed" for p in (FOUR, FIXED))
    protected = json.loads((FIXED / "ORIGINAL_FILES_SHA256.json").read_text())
    protected.update({str(p): sha(p) for folder in (FOUR / "results", FIXED / "results") for p in folder.rglob("*") if p.is_file()})
    assert all(sha(path) == expected for path, expected in protected.items())
    write(ROOT / "ORIGINAL_FILES_SHA256.json", protected)
    write(ROOT / "DEPLOYMENT_MANIFEST.json", {
        "code_commit": "3f07228d39d481da8fb75f0c1b2504f1d25d6966",
        "code_sha256": {p.relative_to(CODE).as_posix(): sha(p) for p in CODE.rglob("*")
                        if p.is_file() and p.suffix in (".py", ".json", ".md")},
        "frozen_original_code_sha256": {name: sha(CODE / name) for name in frozen_names},
        "archive_sha256": sha(ROOT / "code.tar"), "historical_four_root": str(FOUR),
        "historical_fixed_root": str(FIXED), "strong_data_root": str(STRONG),
        "environment_reused": True, "python_executable": sys.executable, "parallel_workers": 4,
        "protocol": json.loads((CODE / "configs/five_models_extra_seeds_v1.json").read_text())})
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_five_models_extra_seeds.py",
        "tests/test_fixed_half_fusion.py", "tests/test_four_models_expanded_val.py", "tests/test_v5_nahsf.py",
        "tests/test_v5_experiment.py", "tests/test_v2_paifilter.py", "tests/test_v2_experiment.py"]
    done = subprocess.run(command, cwd=CODE, env=os.environ.copy(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(done.stdout, flush=True)
    assert done.returncode == 0
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import v2_experiment as data
    import five_models_extra_seeds as experiment
    from v2_paifilter import OriginalCNN, paired_cross_entropy
    from v1_noise_consistency import fix_randomness
    tr, tm, y = data.source_data("train")
    vr, vm, _ = data.source_data("val")
    assert (len(tr), len(vr), len(tm), len(vm)) == (34, 15, 738, 302)
    fixed = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5), materialized_root=STRONG / "generated/source")
    regenerated = data.fixed_noisy_validation(vr, vm, noise_levels=(0, 5))
    assert all(np.array_equal(fixed[k], regenerated[k]) for k in (0, 5))
    clean = data.raw_cycles(tr, tm)
    initial, noise_sha, gpu = {}, {}, {}
    for seed in experiment.NEW_SEEDS:
        noisy = data.fresh_noisy_source(tr, tm, seed, 1)
        # 数据随机数不依赖torch/model随机流，重复生成应逐字节相同。
        assert np.array_equal(noisy, data.fresh_noisy_source(tr, tm, seed, 1))
        noise_sha[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        fix_randomness(seed)
        original = OriginalCNN()
        expected = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in original.parameters())).hexdigest()
        initial[str(seed)] = {}
        initial_logits = {}
        for method in experiment.METHODS:
            fix_randomness(seed)
            model = experiment.make_model(method).cuda().eval()
            initial[str(seed)][method] = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
            assert initial[str(seed)][method] == expected
            inputs = torch.from_numpy(clean[:32]).cuda()
            if method in ("nahsf", "fixed_half"):
                with torch.no_grad():
                    initial_logits[method] = model(inputs).cpu()
            model.train()
            loss = paired_cross_entropy(model, inputs, torch.from_numpy(noisy[:32]).cuda(), torch.from_numpy(y[:32]).cuda())
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            item = {"real_cuda_loss": float(loss.detach().cpu()), "finite_gradients": True,
                    "parameter_counts": __import__("four_models_expanded_val").parameter_counts(model)}
            if method == "nahsf":
                gradient = model.shaping.gate[-1].bias.grad
                assert gradient is not None and torch.isfinite(gradient).all() and float(gradient.abs().max()) > 0
                item["gate_bias_grad_max"] = float(gradient.abs().max())
            gpu[f"{method}/seed{seed}"] = item
            torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.).step()
            del model
        assert torch.equal(initial_logits["nahsf"], initial_logits["fixed_half"])
    assert len(set(noise_sha.values())) == 3
    write(ROOT / "PREFLIGHT.json", {"status": "passed", "unit_test_command": command, "unit_test_output": done.stdout,
          "initial_cnn_sha256": initial, "first_epoch_noise_sha256": noise_sha,
          "fixed_val0_5_matches_original_rng": True, "initial_v5_fixed_logits_exactly_equal": True,
          "real_cuda_checks": gpu, "source_train_records": 34, "source_val_records": 15,
          "source_train_cycles": 738, "source_val_cycles": 302,
          "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__)})
    print("FIVE_MODEL_EXTRA_SEEDS_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
