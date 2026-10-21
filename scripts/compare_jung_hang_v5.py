"""V5 paper-aligned five-phase input builder; anchors always track ground truth."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from jung_hang_models_v5 import (
    JungModifiedCNN, HangWCNN, forward_two_cycle_indices,
    jung_window, hang_trajectory, hang_trajectory_reference_scaled,
)

NUM_CLASSES = 6


def build_inputs(method: str, meta: list[dict], raw: dict, config: dict):
    """Return (input_array, label_indices) with split-local two-cycle windows."""
    if method == "jung":
        windows = forward_two_cycle_indices(meta)
    elif method == "hang":
        windows = [(i, float(row["start"]), float(row["end"]))
                   for i, row in enumerate(meta)]
    else:
        raise ValueError("unknown model " + str(method))
    values = []
    for i, start, end in windows:
        row = meta[i]
        rid = row["record_id"]
        if rid not in raw:
            raise KeyError("missing original record " + str(rid))
        time, currents, _ = raw[rid]
        if method == "jung":
            value = jung_window(time, currents, start, end, points=256)
        else:
            mode = config.get("hang_normalization_mode", "pointwise_eq6")
            if mode not in ("pointwise_eq6", "reference_radius_figures"):
                raise ValueError("Unsupported Hang normalization: " + str(mode))
            if mode == "pointwise_eq6":
                value = hang_trajectory(
                    time, currents, start, end, points=config["hang_points"],
                    size=config["hang_image_size"],
                    phase_angles_deg=tuple(config["hang_phase_angles_deg"]),
                    quiet_ratio=config["hang_quiet_ratio"],
                )
            else:
                value = hang_trajectory_reference_scaled(
                    time, currents, start, end, points=config["hang_points"],
                    size=config["hang_image_size"],
                    phase_angles_deg=tuple(config["hang_phase_angles_deg"]),
                )
        values.append(value)
    array = np.stack(values).astype(np.float32)
    indices = np.asarray([w[0] for w in windows], dtype=np.int64)
    expected_shape = (len(indices), 5, 256) if method == "jung" else (
        len(indices), 1, config["hang_image_size"], config["hang_image_size"]
    )
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise ValueError(f"invalid input shape or values: {array.shape}, {expected_shape}")
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("duplicate label anchor")
    return array, indices


def get_model(method: str, config: dict) -> nn.Module:
    if method == "jung":
        return JungModifiedCNN(in_channels=5, num_classes=NUM_CLASSES)
    if method == "hang":
        return HangWCNN(
            in_channels=1, num_classes=NUM_CLASSES,
            image_size=config["hang_image_size"],
        )
    raise ValueError("unknown model " + str(method))


@torch.no_grad()
def smoke(config: dict):
    j = get_model("jung", config)
    h = get_model("hang", config)
    a = tuple(j(torch.randn(2, 5, 256)).shape)
    b = tuple(h(torch.randn(2, 1, config["hang_image_size"], config["hang_image_size"])).shape)
    assert a == b == (2, NUM_CLASSES)
    return {"jung": a, "hang": b}
