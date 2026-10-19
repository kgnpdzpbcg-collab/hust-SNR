"""Configurable inputs for controlled V4 Jung/Hang fidelity checks."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch
from torch import nn
from jung_hang_models_v4 import JungModifiedCNN, HangWCNN, jung_input, hang_vsd_image, add_noise

NUM_CLASSES = 6


def _two_cycle_bounds(meta):
    """Only join adjacent cycles from same source record and SAME fault label.
    
    If the split has no adjacent same-label neighbour, fail rather than borrow
    a cycle from a validation gap or a different fault state.
    """
    groups = defaultdict(list)
    for i, row in enumerate(meta):
        groups[(row["record_id"], row["label"])].append((i, row))
    result = {}
    for values in groups.values():
        values.sort(key=lambda q: q[1]["start"])
        for j, (index, row) in enumerate(values):
            width = row["end"] - row["start"]
            if width <= 0:
                raise ValueError("Invalid cycle interval")
            candidates = []
            if j > 0:
                previous = values[j - 1][1]
                if abs(previous["end"] - row["start"]) <= 0.05 * width:
                    candidates.append((previous["start"], row["end"]))
            if j + 1 < len(values):
                nxt = values[j + 1][1]
                if abs(nxt["start"] - row["end"]) <= 0.05 * width:
                    candidates.append((row["start"], nxt["end"]))
            if not candidates:
                raise ValueError(
                    f"Two-cycle input unavailable within this split/label: "
                    f"{row['record_id']} label={row['label']} start={row['start']}. "
                    "Do not cross gap/fault boundaries; use one_cycle or revise manifest."
                )
            result[index] = candidates[0]
    return result


def build_inputs(method: str, meta: list[dict], raw: dict, config: dict) -> np.ndarray:
    values = []
    double = method == "jung" and config["jung_cycles"] == 2
    bounds = _two_cycle_bounds(meta) if double else None
    for i, row in enumerate(meta):
        rid = row["record_id"]
        if rid not in raw:
            raise KeyError(f"Missing original record {rid}")
        time, currents, _ = raw[rid]
        start, end = bounds[i] if double else (row["start"], row["end"])
        if method == "jung":
            value = jung_input(time, currents, start, end,
                               points=128 * config["jung_cycles"],
                               scale_mode=config["jung_scale"])
        elif method == "hang":
            value = hang_vsd_image(time, currents, start, end,
                                   points=128, image_size=40,
                                   planes=config["hang_planes"],
                                   center_mode=config["hang_center"],
                                   normalization=config["hang_normalization"])
        else:
            raise ValueError(f"Unknown model: {method}")
        values.append(value)
    result = np.stack(values).astype(np.float32)
    expected = ((len(meta), 5, 128 * config["jung_cycles"]) if method == "jung"
                else (len(meta), 2, 40, 40))
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"Input shape/values incorrect: {result.shape} != {expected}")
    return result


def get_model(method: str, config: dict) -> nn.Module:
    if method == "jung":
        return JungModifiedCNN(in_channels=5, num_classes=NUM_CLASSES)
    if method == "hang":
        return HangWCNN(in_channels=2, num_classes=NUM_CLASSES, image_size=40)
    raise ValueError(f"Unknown model: {method}")


@torch.no_grad()
def smoke(config: dict) -> dict:
    torch.manual_seed(42)
    j, h = get_model("jung", config), get_model("hang", config)
    sj = tuple(j(torch.randn(3, 5, 128 * config["jung_cycles"])).shape)
    sh = tuple(h(torch.randn(3, 2, 40, 40)).shape)
    assert sj == sh == (3, 6)
    return {"jung_output": sj, "hang_output": sh,
            "jung_parameters": sum(p.numel() for p in j.parameters()),
            "hang_parameters": sum(p.numel() for p in h.parameters())}
