"""补齐服务器代码快照的历史读取路径，仅建立链接，不读取实测波形。"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2]
CODE = ROOT / "code"
PREVIOUS = BASE / "experiments/prototypes_a1_b1/dc8cb36/code"
(CODE / "data/target_dev").mkdir(parents=True)
links = {}
for name, target in (
    ("data/source", PREVIOUS / "data/source"),
    ("data/target_dev/clean", PREVIOUS / "data/target_dev/clean"),
):
    link = CODE / name
    assert not link.exists() and target.is_dir()
    link.symlink_to(target.resolve(), target_is_directory=True)
    links[name] = str(target.resolve())
(ROOT / "FROZEN_READER_LINKS.json").write_text(json.dumps({"links": links,
    "source_v2_root": str((CODE / "outputs/source_cycle_split_v2").resolve()),
    "target_waveforms_read": False, "model_training_started": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("FROZEN_READER_PATHS_LINKED_WITHOUT_READING_WAVEFORMS")
