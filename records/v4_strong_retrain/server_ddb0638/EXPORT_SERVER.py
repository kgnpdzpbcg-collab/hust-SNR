"""仅打包配置、指标、标签预测和日志；排除MAT、电流数组、权重及环境。"""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
files = [p for p in (root / "results").rglob("*") if p.is_file() and p.suffix in (".json", ".csv")]
files += list((root / "logs").glob("*.json")) + list((root / "logs").glob("*.log"))
files += [root / name for name in ("RUN_STATUS.json", "PREFLIGHT.json", "DEPLOYMENT_MANIFEST.json",
          "ORIGINAL_FILES_SHA256.json", "CONTROLLER_PID.json", "SETUP_AND_PREFLIGHT.py", "RUN_SERVER.py",
          "LAUNCH.py", "VERIFY_AND_SUMMARIZE.py", "CHECK_SHARED_METADATA.py", "EXPORT_SERVER.py", "controller.log", "summary.log")]
files += [root / "code/configs/v4_strong_noise_v1.json"]
assert json.loads((root / "results/VERIFICATION.json").read_text())["status"] == "passed"
manifest = {"status": "passed", "files": {str(p.relative_to(root)): {"bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},
            "raw_current_uploaded": False, "weights_uploaded": False, "environment_uploaded": False}
path = root / "RESULTS_EXPORT_MANIFEST.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
with tarfile.open(root / "server_records.tar.gz", "w:gz") as archive:
    for file in files + [path]:
        archive.add(file, arcname=str(file.relative_to(root)), recursive=False)
print("EXPORT_PASSED", len(files), "files", (root / "server_records.tar.gz").stat().st_size, "bytes")
