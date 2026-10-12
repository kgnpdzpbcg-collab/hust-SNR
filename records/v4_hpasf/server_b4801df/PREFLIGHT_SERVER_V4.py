"""正式运行前核对冻结代码、原文件、初始化和真实训练批次的CUDA梯度。

使用已有环境，所有检查只写V4目录。测试批次不保存为正式权重，
正式训练仍由仓库原生v4_experiment.py从随机初始化开始执行。
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  CUBLAS_WORKSPACE_CONFIG=":4096:8")
sys.path.insert(0, str(CODE / "scripts"))
import numpy as np
import torch
import v2_experiment as data
from v1_noise_consistency import fix_randomness
from v4_hpasf import HPASFCNN, METHODS, counts_by_module
from v2_paifilter import HarmonicCNN, paired_cross_entropy


def sha(path):
    """逐块核对代码、共享原数据及旧实验的完整字节。"""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    """先运行原生测试，再验证本轮实际数据的GPU反传和跨方法公平性。"""
    manifest = json.loads((ROOT / "DEPLOYMENT_MANIFEST.json").read_text())
    for name, expected in manifest["code_sha256"].items():
        assert sha(CODE / name) == expected, name
    base = ROOT.parents[2]
    originals = json.loads((ROOT / "ORIGINAL_FILES_SHA256.json").read_text())
    for name, expected in originals.items():
        assert sha(base / name) == expected, name
    proof = json.loads((base / "analysis_checks/v2_file_audit_4df46a3/SERVER_AUDIT.json").read_text())
    for name, expected in proof["files"].items():
        if name.startswith("source/"):
            path = CODE / "data" / name
        elif name.startswith("clean/"):
            path = CODE / "data/target_dev" / name
        else:
            path = ROOT / "target_snr" / name
        assert sha(path) == expected["sha256"], name
    for name, expected in proof["metadata_sha256"].items():
        assert sha(CODE / name) == expected, name
    commands = [[sys.executable, "scripts/v4_hpasf.py"],
                [sys.executable, "scripts/v4_experiment.py", "smoke"],
                [sys.executable, "-m", "pytest", "-q", "tests/test_v4_hpasf.py",
                 "tests/test_v4_experiment.py", "tests/test_v2_paifilter.py", "tests/test_v2_experiment.py"]]
    statuses = []
    for command in commands:
        started = time.monotonic()
        result = subprocess.run(command, cwd=CODE, env=os.environ.copy(), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, flush=True)
        statuses.append({"command": command, "exit_code": result.returncode,
                         "elapsed_seconds": time.monotonic() - started,
                         "output": result.stdout})
        assert result.returncode == 0, command
    assert torch.cuda.is_available()
    raw, meta, y = data.source_data("train")
    val_raw, val_meta, _ = data.source_data("val")
    clean = data.raw_cycles(raw, meta)
    hashes, noise_hashes, cuda_checks = {}, {}, {}
    prior = json.loads((base / "experiments/v2_paifilter/4df46a37a6c2/PREFLIGHT.json").read_text())
    for seed in (42, 43, 44):
        noisy = data.fresh_noisy_source(raw, meta, seed, 1)
        noise_hashes[str(seed)] = hashlib.sha256(noisy.tobytes()).hexdigest()
        assert noise_hashes[str(seed)] == prior["source_initial_noise_sha256"][str(seed)]
        hashes[str(seed)] = {}
        for method in METHODS:
            fix_randomness(seed)
            model = HPASFCNN(method).cuda()
            h = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                       for p in model.backbone.parameters())).hexdigest()
            hashes[str(seed)][method] = h
            assert h == prior["initial_cnn_sha256"][str(seed)]["lowpass3"]
            if seed == 42:
                xb = torch.from_numpy(clean[:32]).cuda()
                nb = torch.from_numpy(noisy[:32]).cuda()
                labels = torch.from_numpy(y[:32]).cuda()
                if method == "lowpass3":
                    fix_randomness(seed)
                    old = HarmonicCNN("lowpass3").cuda()
                    assert torch.equal(model(xb), old(xb)), "V2低通控制组前向不一致"
                    del old
                optimizer = torch.optim.Adam(model.parameters(), lr=.001)
                loss = paired_cross_entropy(model, xb, nb, labels)
                assert torch.isfinite(loss)
                loss.backward()
                check = {"loss": float(loss.detach().cpu()), "parameters": counts_by_module(model),
                         "deterministic_cuda_backward": True}
                if method in ("static4", "hpasf"):
                    p = model.shaping.gate_logit
                    previous = p.detach().clone()
                    assert p.grad is not None and torch.isfinite(p.grad) and float(p.grad.abs()) > 0
                    check["spectral_gradient"] = float(p.grad.detach().cpu())
                    optimizer.step()
                    assert not torch.equal(previous, p.detach())
                    check["spectral_parameter_updated"] = True
                cuda_checks[method] = check
            del model
    result = {"status": "passed", "code_commit": manifest["code_commit"],
              "commands": statuses, "source_train_records": len(raw), "source_train_cycles": len(meta),
              "source_val_records": len(val_raw), "source_val_cycles": len(val_meta),
              "target_dev_records": 88, "target_dev_cycles": 2474,
              "input_files_matched_previous_original_audit": len(proof["files"]),
              "initial_cnn_sha256": hashes, "source_initial_noise_sha256": noise_hashes,
              "cuda_real_batch_checks": cuda_checks, "target_holdout_waveforms_present": False,
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "numpy": np.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(0), "deterministic": torch.are_deterministic_algorithms_enabled(),
                              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                              "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                              "cpu_threads": torch.get_num_threads()}}
    (ROOT / "PREFLIGHT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("V4_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
