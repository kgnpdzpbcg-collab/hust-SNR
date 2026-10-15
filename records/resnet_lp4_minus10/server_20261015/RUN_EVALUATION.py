"""部署独立−10 dB推理入口，复用服务器环境，导出客观记录及预测。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
REF = BASE / "experiments/architecture_suite/5c75bea"
DATA = BASE / "datasets/strong_snr_minus10_v1_20261015"
CODE = ROOT / "code"


def sha(path):
    """逐字节指纹核对代码快照及最终发布文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    """在新目录记录来源、命令和耗时，不修改历史实验。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    """先校验原代码快照，完成实际推理及脚本内核验后才导出元数据。"""
    assert not CODE.exists() and not (ROOT / "results").exists()
    CODE.mkdir()
    with tarfile.open(ROOT / "code.tar") as archive:
        archive.extractall(CODE, filter="data")
    prior = json.loads((REF / "DEPLOYMENT_MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in prior["code_sha256"].items():
        assert sha(REF / "code" / name) == digest, name
    import torch
    write(ROOT / "DEPLOYMENT_MANIFEST.json", {"client_date": "2026-10-15", "evaluation_code_commit": (ROOT / "CODE_COMMIT.txt").read_text(encoding="utf-8-sig").strip(),
          "reference_training_code_commit": prior["code_commit"], "reference_code_snapshot_verified": True,
          "python": sys.executable, "torch_version": torch.__version__,
          "environment_reused": True, "dependencies_installed": False,
          "reference_experiment": str(REF), "minus10_dataset": str(DATA),
          "code_archive_sha256": sha(ROOT / "code.tar"), "new_result_directory": str(ROOT / "results")})
    command = [sys.executable, "-u", str(CODE / "scripts/evaluate_resnet_lp4_minus10.py"),
               "--reference-code", str(REF / "code"), "--reference-results", str(REF / "results"),
               "--minus10-root", str(DATA),
               "--minus5-root", str(BASE / "datasets/strong_snr_minus5_v1_20261014"),
               "--minus5-results", str(BASE / "experiments/architecture_minus5/20261014/results"),
               "--output", str(ROOT / "results"),
               "--config", str(CODE / "configs/resnet_lp4_minus10_fixed_weights_v1.json")]
    started = time.perf_counter()
    print("START: original 3 resnet_lp4 fixed checkpoints, 88 Dev files, -10 dB", flush=True)
    with (ROOT / "evaluation.log").open("w", encoding="utf-8") as stream:
        finished = subprocess.run(command, cwd=REF / "code", stdout=stream, stderr=subprocess.STDOUT)
    write(ROOT / "EVALUATION_COMMAND.json", {"command": command, "cwd": str(REF / "code"),
          "exit_code": finished.returncode, "elapsed_seconds": time.perf_counter() - started})
    print((ROOT / "evaluation.log").read_text(encoding="utf-8"), flush=True)
    assert finished.returncode == 0, "推理或核验失败，停止发布"
    files = {"DEPLOYMENT_MANIFEST.json": ROOT / "DEPLOYMENT_MANIFEST.json",
             "EVALUATION_COMMAND.json": ROOT / "EVALUATION_COMMAND.json", "evaluation.log": ROOT / "evaluation.log",
             "RUN_EVALUATION.py": Path(__file__), "configs/resnet_lp4_minus10_fixed_weights_v1.json": CODE / "configs/resnet_lp4_minus10_fixed_weights_v1.json",
             "scripts/evaluate_resnet_lp4_minus10.py": CODE / "scripts/evaluate_resnet_lp4_minus10.py"}
    files["scripts/evaluate_arch_suite_minus5.py"] = CODE / "scripts/evaluate_arch_suite_minus5.py"
    files.update({str(path.relative_to(ROOT)): path for path in (ROOT / "results").rglob("*") if path.is_file()})
    assert not any(path.suffix in (".mat", ".pt", ".pth", ".npz") for path in files.values())
    write(ROOT / "RESULT_EXPORT_MANIFEST.json", {"files": {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())},
          "raw_waveforms_included": False, "model_weights_included": False})
    files["RESULT_EXPORT_MANIFEST.json"] = ROOT / "RESULT_EXPORT_MANIFEST.json"
    with tarfile.open(ROOT / "resnet_lp4_minus10_records.tar.gz", "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name)
    print("ALL_COMPLETE: isolated minus10 evaluation and export verified", flush=True)


if __name__ == "__main__":
    main()
