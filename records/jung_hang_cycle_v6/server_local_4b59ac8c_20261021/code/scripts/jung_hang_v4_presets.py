"""Create *separate*, source-only Jung/Hang V4 control experiment configs.

Each preset changes one input/training choice relative to its own reference.
This command writes configs; it does NOT train models or inspect target data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "configs/jung_hang_cycle_v4_v1.json"
PRESETS = {
    "jung_j0_single_shared": ("jung", {}),
    "jung_j0_double_shared": ("jung", {"jung_cycles": 2}),
    "jung_j1_per_phase": ("jung", {"jung_scale": "per_phase"}),
    "jung_j2_quiet_gate": ("jung", {"training_noise": "quiet"}),
    "hang_h0_dual_no_center": ("hang", {}),
    "hang_h0_centered": ("hang", {"hang_center": "phase_mean"}),
    "hang_h1_fundamental_only": ("hang", {"hang_planes": "fundamental"}),
    "hang_h2_instantaneous": ("hang", {"hang_normalization": "instantaneous"}),
    "hang_h3_clean_only": ("hang", {"hang_training_view": "clean"}),
}


def generate(out: Path) -> list[Path]:
    base = json.loads(BASE.read_text(encoding="utf8"))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (method, changes) in PRESETS.items():
        c = {**base, "models": [method],
             "protocol": base["protocol"] + ":" + name, **changes}
        path = out / (name + ".json")
        path.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/jung_hang_v4_configurations"))
    args = parser.parse_args()
    for path in generate(args.output):
        print(path)
