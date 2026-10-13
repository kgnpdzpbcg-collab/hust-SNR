"""导出新指标、历史引用及六种子汇总，不上传原始电流、权重或环境。"""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
assert json.loads((root / "results/VERIFICATION.json").read_text())["status"] == "passed"
files = [p for p in (root / "results").rglob("*") if p.is_file() and p.suffix in (".json", ".csv")]
files += list((root / "logs").glob("*.log")) + list((root / "logs").glob("*.json"))
files += [root / name for name in ("DEPLOYMENT_MANIFEST.json", "PREFLIGHT.json", "ORIGINAL_FILES_SHA256.json",
    "RUN_STATUS.json", "CONTROLLER_PID.json", "SETUP_AND_PREFLIGHT.py", "RUN_SERVER.py", "LAUNCH.py",
    "VERIFY_AND_SUMMARIZE.py", "EXPORT_SERVER.py", "controller.log", "summary.log")]
files += [root / "code/configs/five_models_extra_seeds_v1.json"]
manifest = {"status": "passed", "files": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size,
    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},
    "raw_current_uploaded": False, "weights_uploaded": False, "environment_uploaded": False}
p = root / "RESULTS_EXPORT_MANIFEST.json"
p.write_text(json.dumps(manifest, indent=2) + "\n")
with tarfile.open(root / "server_records.tar.gz", "w:gz") as archive:
    for file in files + [p]:
        archive.add(file, arcname=file.relative_to(root).as_posix(), recursive=False)
print("FIVE_MODEL_EXTRA_SEEDS_EXPORT_PASSED", len(files), "files")
