"""V3 input adapters and model factory for the HUST five-phase IM-AA task.

The cycle-V3 runner uses these functions while preserving the earlier V1/V2
experiment code and records. This module intentionally exposes only the shared
model/input API needed by the frozen-cycle runner.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from jung_hang_models_v3 import (
    HangWCNN,
    JungModifiedCNN,
    current_window_to_jung_input,
    current_window_to_normalized_vsd_image,
)

NUM_CLASSES = 6


def build_inputs(method: str, meta: list[dict], raw: dict) -> np.ndarray:
    """Create V3 inputs from raw records, not legacy per-phase-normalized caches."""
    values = []
    for row in meta:
        rid = row["record_id"]
        if rid not in raw:
            raise KeyError(f"Raw record {rid} required by frozen window metadata is missing")
        time, currents, _ = raw[rid]
        if method == "jung":
            value = current_window_to_jung_input(
                time, currents, row["start"], row["end"], points=128
            )
        elif method == "hang":
            value = current_window_to_normalized_vsd_image(
                time, currents, row["start"], row["end"], points=128, image_size=40
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        values.append(value)
    result = np.stack(values).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Non-finite {method} inputs")
    expected = (len(meta), 5, 128) if method == "jung" else (len(meta), 1, 40, 40)
    if result.shape != expected:
        raise RuntimeError(f"Unexpected {method} input shape {result.shape}; expected {expected}")
    return result


def get_model(method: str) -> nn.Module:
    if method == "jung":
        return JungModifiedCNN(in_channels=5, num_classes=NUM_CLASSES)
    if method == "hang":
        return HangWCNN(in_channels=1, num_classes=NUM_CLASSES, image_size=40)
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def smoke() -> dict:
    """Synthetic-only shape and finite-output check; does not read any dataset."""
    torch.manual_seed(42)
    jung = get_model("jung")
    xj = torch.randn(4, 5, 128)
    yj = jung(xj)
    hang = get_model("hang")
    xh = torch.randn(4, 1, 40, 40)
    yh = hang(xh)
    assert yj.shape == (4, 6) and torch.isfinite(yj).all()
    assert yh.shape == (4, 6) and torch.isfinite(yh).all()
    return {
        "jung_parameters": sum(p.numel() for p in jung.parameters()),
        "hang_parameters": sum(p.numel() for p in hang.parameters()),
        "jung_output_shape": list(yj.shape),
        "hang_output_shape": list(yh.shape),
    }
