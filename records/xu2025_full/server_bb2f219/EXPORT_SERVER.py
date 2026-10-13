"""导出Xu-Full指标、训练分量和来源记录；不发布原始波形/模型权重/环境。"""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
assert json.loads((root/"results/matched/reports/VERIFICATION.json").read_text())["status"]=="passed"
files=[p for p in (root/"results").rglob("*") if p.is_file() and p.suffix in (".json",".csv")]
files += list((root/"logs").glob("*.log"))+list((root/"logs").glob("*.json"))
files += [root/name for name in ("DEPLOYMENT_MANIFEST.json","PREFLIGHT.json","ORIGINAL_FILES_SHA256.json","RUN_STATUS.json",
          "CONTROLLER_PID.json","SETUP_AND_PREFLIGHT.py","RUN_SERVER.py","LAUNCH.py","VERIFY_AND_SUMMARIZE.py",
          "EXPORT_SERVER.py","controller.log","summary.log")]
files += [root/"code/configs/xu2025_full_v1.json"]
manifest={"status":"passed","files":{p.relative_to(root).as_posix():{"bytes":p.stat().st_size,
           "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},
          "raw_current_uploaded":False,"weights_uploaded":False,"environment_uploaded":False}
path=root/"RESULTS_EXPORT_MANIFEST.json"
path.write_text(json.dumps(manifest,indent=2)+"\n")
with tarfile.open(root/"server_records.tar.gz","w:gz") as archive:
    for p in files+[path]:archive.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print("XU_FULL_EXPORT_PASSED",len(files),"files")
