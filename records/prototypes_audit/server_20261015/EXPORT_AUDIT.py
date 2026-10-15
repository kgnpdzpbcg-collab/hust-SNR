"""导出本轮固定权重审计的代码与元数据，不包含电流数组或权重。"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import tarfile

root = Path(__file__).resolve().parent


def sha(path):
    """核对实际交付文件字节，防止传输或换行改变结果。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    """只创建新的审计来源记录。"""
    assert not path.exists()
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


save(root / "RUN_MANIFEST.json", {
    "client_date": "2026-10-15", "audit_code_commit": "da0e71c00f7c07253e1466182ad599d1be5fbdf0",
    "reference_algorithm_code_commit": "c0b30c9094700cc45e31ebc558d7707dc6fe46fa",
    "python": sys.executable, "python_version": platform.python_version(), "environment_reused": True,
    "models_trained": 0, "dependencies_installed": False,
    "command": [sys.executable, "-u", str(root / "audit_a0_b0_fixed_weights.py"),
        "--reference-root", "/home/wang/projects/会议论文/hust/experiments/prototypes_a0_b0/c0b30c9",
        "--output", str(root / "results")],
    "exit_code": 0, "final_holdout_accessed": False})
verification = json.loads((root / "results/VERIFICATION.json").read_text())
assert verification["status"] == "passed"
files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()
         and p.suffix in (".py", ".json", ".csv") and "__pycache__" not in p.parts}
assert not any(p.suffix in (".mat", ".pt", ".npz") for p in files.values())
save(root / "EXPORT_MANIFEST.json", {"files": {name: {"sha256": sha(p), "bytes": p.stat().st_size}
      for name, p in sorted(files.items())}, "raw_waveforms_included": False, "weights_included": False})
files["EXPORT_MANIFEST.json"] = root / "EXPORT_MANIFEST.json"
with tarfile.open(root / "audit_records.tar.gz", "w:gz") as archive:
    for name, path in sorted(files.items()):
        archive.add(path, arcname=name)
print("AUDIT_METADATA_EXPORTED", len(files))
