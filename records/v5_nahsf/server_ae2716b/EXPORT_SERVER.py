"""导出V5指标、预测标签、门控统计和日志，不导出电流、MAT或权重。"""
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
manifest = {"status": "passed", "files": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},
            "raw_currents_uploaded": False, "weights_uploaded": False, "environment_uploaded": False}
path = root / "RESULTS_EXPORT_MANIFEST.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
with tarfile.open(root / "server_records.tar.gz", "w:gz") as archive:
    for p in files + [path]:
        archive.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
print("V5_EXPORT_PASSED", len(files), "files")
