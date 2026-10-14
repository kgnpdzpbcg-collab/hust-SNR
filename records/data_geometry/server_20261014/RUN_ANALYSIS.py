"""在原服务器环境执行只读数据几何分析并导出元数据，留存实际命令。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]


def sha(path):
    """核对导出字节，不发布任何波形、缓存或模型权重。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """测试通过后只执行数据描述统计；输出独立，不修改旧实验。"""
    assert not (ROOT / "results").exists()
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    with (ROOT / "unit_tests.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "test_data_geometry.py")],
                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    assert completed.returncode == 0, (ROOT / "unit_tests.log").read_text(encoding="utf-8")
    command = [sys.executable, "-u", str(ROOT / "analyze_data_geometry.py"),
               "--reference-experiment", str(BASE / "experiments/architecture_suite/5c75bea"),
               "--difficulty-experiment", str(BASE / "experiments/lp4_task_difficulty/20261014"),
               "--strong-root", str(BASE / "datasets/strong_snr_v1_20261012"),
               "--minus5-root", str(BASE / "datasets/strong_snr_minus5_v1_20261014"), "--output", str(ROOT / "results")]
    started = time.perf_counter()
    with (ROOT / "analysis.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    record = {"client_date": "2026-10-14", "code_commit": "33c812c3df59b96950fa39990facf6adf8a2f98b",
              "command": command, "cwd": str(ROOT), "python_environment_reused": True, "exit_code": completed.returncode,
              "elapsed_seconds": time.perf_counter() - started, "classifiers_trained": 0, "classifier_weights_loaded": False}
    (ROOT / "ANALYSIS_COMMAND.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print((ROOT / "analysis.log").read_text(encoding="utf-8"), flush=True)
    assert completed.returncode == 0
    files = {path.name: path for path in ROOT.iterdir() if path.is_file() and path.suffix in (".py", ".json", ".log")}
    files.update({str(path.relative_to(ROOT)): path for path in (ROOT / "results").iterdir() if path.is_file()})
    manifest = ROOT / "EXPORT_MANIFEST.json"
    manifest.write_text(json.dumps({"files": {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())},
                        "raw_waveforms_or_feature_arrays_included": False}, indent=2) + "\n", encoding="utf-8")
    files[manifest.name] = manifest
    with tarfile.open(ROOT / "data_geometry_records.tar.gz", "w:gz") as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name)
    print("ALL_COMPLETE: data-only analysis verified and metadata exported", flush=True)


if __name__ == "__main__":
    main()
