"""V5部署及正式训练前检查：原V4数据/数值协议、真实CUDA梯度和旧文件。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
OLD = BASE / "experiments/v4_hpasf/b4801df"
PRESSURE = BASE / "experiments/v4_strong_snr/84f6757"
RETRAIN = BASE / "experiments/v4_strong_retrain/ddb0638"
STRONG = BASE / "datasets/strong_snr_v1_20261012"
CODE = ROOT / "code"


def sha(path):
    """分块核对实际文件字节，原数据和所有旧权重/成绩保持只读。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    """本次部署和检查元数据仅写入独立V5目录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    """检查通过后才允许从随机初始化开始训练正式三种子模型。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    for path in (CODE / "data/target_dev", CODE / "outputs"):
        path.mkdir(parents=True)
    (CODE / "data/source").symlink_to(OLD / "code/data/source", target_is_directory=True)
    (CODE / "data/target_dev/clean").symlink_to(OLD / "code/data/target_dev/clean", target_is_directory=True)
    (CODE / "outputs/gate0_v1").symlink_to(OLD / "code/outputs/gate0_v1", target_is_directory=True)
    (ROOT / "target_snr").symlink_to(OLD / "target_snr", target_is_directory=True)
    # V5测试回读真实已发布历史文件，链接与汇总使用的实际旧结果完全相同。
    for name, original in (("v4_hpasf/server_b4801df", OLD), ("v4_strong_snr/server_84f6757", PRESSURE)):
        parent = CODE / "records" / name
        parent.mkdir(parents=True)
        (parent / "results").symlink_to(original / "results", target_is_directory=True)
    original_deployment = json.loads((OLD / "DEPLOYMENT_MANIFEST.json").read_text())
    for name in ("scripts/v2_paifilter.py", "scripts/v1_noise_consistency.py", "scripts/prepare.py", "configs/splits_gate0_v1.json"):
        assert sha(CODE / name) == original_deployment["code_sha256"][name], name
    # 同时保护旧V4重训结果，以免不同实验输出相互覆盖。
    paths = [p for directory in (OLD / "results", PRESSURE / "results", RETRAIN / "results")
             for p in directory.rglob("*") if p.is_file()]
    original_gate0 = json.loads((OLD / "ORIGINAL_FILES_SHA256.json").read_text())
    for name, expected in original_gate0.items():
        assert sha(BASE / name) == expected
        paths.append(BASE / name)
    paths += [p for p in (CODE / "data/source").glob("*.mat")]
    paths += [p for p in (CODE / "data/target_dev/clean").glob("*.mat")]
    paths += [p for level in (10, 15, 20, 25, 30, 35, 40) for p in (ROOT / f"target_snr/SNR_{level}").glob("*.mat")]
    generated = json.loads((STRONG / "generated/GENERATION_MANIFEST.json").read_text())
    assert generated["status"] == "passed" and json.loads((STRONG / "VERIFICATION.json").read_text())["status"] == "passed"
    for row in generated["files"]:
        if row["domain"] == "experimental" and row["partition"] == "dev":
            path = STRONG / "generated" / row["file"]
            assert sha(path) == row["file_sha256"]
            paths.append(path)
    write(ROOT / "ORIGINAL_FILES_SHA256.json", {str(p): sha(p) for p in paths})
    code_sha = {str(p.relative_to(CODE)): sha(p) for p in CODE.rglob("*")
                if p.is_file() and p.suffix in (".py", ".json", ".md")}
    write(ROOT / "DEPLOYMENT_MANIFEST.json", {"code_commit": "ae2716bbcd3dd23a959696b486103ac50efa087d",
          "upstream_commit": "90d268c3fb0ff84080d9d3918d41f8daef875495", "branch": "research/na-hsf-v5",
          "execution_fixes": ["V4 legacy stress snr_db schema reader", "V4 deterministic evaluation and TF32 settings"],
          "code_sha256": code_sha, "archive_sha256": sha(ROOT / "code.tar"), "original_v4": str(OLD),
          "original_pressure_test": str(PRESSURE), "strong_data_root": str(STRONG), "environment_reused": True,
          "seeds": [42, 43, 44], "epochs": 50, "batch_size": 32, "learning_rate": .001,
          "train_noise_db": [10, 15, 20, 25, 30, 35, 40], "val_noise_db": [10, 20, 30],
          "target_conditions": ["clean", 10, 15, 20, 25, 30, 35, 40, 5, 0], "parallel_seed_workers": 3,
          "final_holdout_evaluated": False})
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    commands = [[sys.executable, "scripts/v5_nahsf.py"], [sys.executable, "scripts/v5_experiment.py", "smoke"],
                [sys.executable, "-m", "pytest", "-q", "tests/test_v5_nahsf.py", "tests/test_v5_experiment.py",
                 "tests/test_v2_paifilter.py", "tests/test_v2_experiment.py"]]
    tests = []
    for command in commands:
        completed = subprocess.run(command, cwd=CODE, env=os.environ.copy(), text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout, flush=True)
        tests.append({"command": command, "exit_code": completed.returncode, "output": completed.stdout})
        assert completed.returncode == 0
    sys.path.insert(0, str(CODE / "scripts"))
    import numpy as np
    import torch
    import v2_experiment as data
    from v5_nahsf import NAHSFCNN, parameter_counts
    from v1_noise_consistency import fix_randomness
    from v2_paifilter import paired_cross_entropy
    prior = json.loads((OLD / "PREFLIGHT.json").read_text())
    raw, meta, labels = data.source_data("train")
    vr, vm, _ = data.source_data("val")
    assert (len(raw), len(vr), len(meta), len(vm)) == (34, 15, 738, 302)
    assert sorted(data.fixed_noisy_validation(vr, vm)) == [10, 20, 30]
    clean = data.raw_cycles(raw, meta)
    initial, noise_hash, gpu = {}, {}, {}
    for seed in (42, 43, 44):
        fix_randomness(seed)
        noisy = data.fresh_noisy_source(raw, meta, seed, 1)
        noise_hash[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noise_hash[str(seed)] == prior["source_initial_noise_sha256"][str(seed)]
        model = NAHSFCNN().cuda()
        initial[str(seed)] = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.backbone.parameters())).hexdigest()
        assert initial[str(seed)] == prior["initial_cnn_sha256"][str(seed)]["lowpass3"]
        assert parameter_counts(model) == {"cnn": 25620, "spectral": 41, "total": 25661}
        xb, nb, y = torch.from_numpy(clean[:32]).cuda(), torch.from_numpy(noisy[:32]).cuda(), torch.from_numpy(labels[:32]).cuda()
        with torch.no_grad():
            alpha = model.shaping(xb, return_gate=True)[1]
            assert torch.equal(alpha, torch.full_like(alpha, .5))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        loss = paired_cross_entropy(model, xb, nb, y)
        loss.backward()
        last = model.shaping.gate[-1]
        assert torch.isfinite(loss) and last.bias.grad is not None and float(last.bias.grad.abs().max()) > 0
        assert torch.isfinite(last.weight.grad).all() and float(last.weight.grad.abs().max()) > 0
        weight_before = last.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(last.weight.detach(), weight_before)
        optimizer.zero_grad(set_to_none=True)
        paired_cross_entropy(model, xb, nb, y).backward()
        first = model.shaping.gate[0].weight.grad
        assert first is not None and torch.isfinite(first).all() and float(first.abs().max()) > 0
        gpu[str(seed)] = {"first_step_loss": float(loss.detach().cpu()), "first_layer_second_step_grad_max": float(first.abs().max()),
                          "last_layer_first_step_grad_max": float(last.bias.grad.abs().max()), "strict_deterministic_cuda": True}
        del model
    write(ROOT / "PREFLIGHT.json", {"status": "passed", "tests": tests, "source_train_records": 34,
          "source_val_records": 15, "train_cycles": 738, "val_cycles": 302, "initial_cnn_sha256": initial,
          "first_epoch_noise_sha256": noise_hash, "noise_matches_original_v4": True, "real_cuda_checks": gpu,
          "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
          "original_dataset_and_historical_files_unchanged": True})
    print("V5_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
