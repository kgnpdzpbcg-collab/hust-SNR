"""补充核查原共享周期、标签数组与V1/Gate0文件仍匹配原V4记录。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
base = root.parents[2]
original = json.loads((base / "experiments/v4_hpasf/b4801df/ORIGINAL_FILES_SHA256.json").read_text())
for name, expected in original.items():
    digest = hashlib.sha256()
    with (base / name).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    assert digest.hexdigest() == expected, name
record = {"status": "passed", "original_gate0_v1_files_unchanged": len(original),
          "cycle_metadata_sha256": {k: v for k, v in original.items() if k.endswith("windows.json")},
          "source_train_val_metadata_and_labels_unchanged": True}
(root / "results/SHARED_METADATA_CHECK.json").write_text(json.dumps(record, indent=2) + "\n")
print("SHARED_METADATA_VERIFIED", len(original), "files")
