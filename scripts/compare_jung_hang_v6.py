"""V6：Jung / Hang 五相输入构建与模型入口。

两个模型都使用**单电周期**窗口，因此训练/验证/实测样本数与其余八个模型
完全一致（源域 962/221，Dev 2474），不需要再为了对齐而裁剪目标队列。
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from jung_hang_models_v6 import (
    JungModifiedCNN, HangWCNN, jung_window, hang_trajectory, record_reference,
)

NUM_CLASSES = 6


def _references(raw, record_ids, reference_s, phase_angles_deg):
    """按记录缓存 (dc, scale)，避免每个窗口重复做一次 FFT 之外的重采样。

    phase_angles_deg 必须与轨迹构造使用同一组角度，否则健康尺度与轨迹
    不在同一坐标系下。
    """
    cache = {}
    for rid in record_ids:
        if rid in cache:
            continue
        time, currents, _ = raw[rid]
        cache[rid] = record_reference(time, currents, reference_s, phase_angles_deg)
    return cache


def build_inputs(method, meta, raw, config):
    """构建模型输入。

    参数：
        method："jung" 或 "hang"。
        meta：窗口元数据列表，每行含 record_id / label / start / end。
        raw：record_id -> (time, currents[T,5], omega)。
        config：V6 冻结配置。

    返回：
        (inputs, indices)
        inputs：Jung 为 [N,5,points]，Hang 为 [N,channels,size,size]
                （channels=2 为基波 + 三次双平面）。
        indices：长度 N，对应 meta 的行号，严格等于 range(N)（单周期窗口
                 不跨分组、不丢锚点），保留该字段是为了与评价记录对齐。

    说明：
        两个模型都是"一个电周期一个样本"，与源划分和 Dev 队列的切法一致，
        因此这里不再构造跨周期窗口，也不做任何锚点裁剪。
    """
    if method not in ("jung", "hang"):
        raise ValueError("unknown model " + str(method))
    if config["jung_cycles"] != 1 or config["hang_cycles"] != 1:
        raise ValueError("V6 只允许单电周期窗口")

    references = _references(
        raw, sorted({row["record_id"] for row in meta}),
        tuple(config["reference_window_s"]),
        tuple(config["hang_phase_angles_deg"]),
    )

    values = []
    indices = []
    for position, row in enumerate(meta):
        rid = row["record_id"]
        if rid not in raw:
            raise KeyError("missing original record " + str(rid))
        time, currents, _ = raw[rid]
        dc, scale, _ = references[rid]
        start, end = float(row["start"]), float(row["end"])
        if method == "jung":
            values.append(jung_window(
                time, currents, start, end, dc,
                points=config["jung_points_per_cycle"],
            ))
        else:
            values.append(hang_trajectory(
                time, currents, start, end, dc, scale,
                points=config["hang_points"],
                size=config["hang_image_size"],
                phase_angles_deg=tuple(config["hang_phase_angles_deg"]),
                planes=config["hang_planes"],
            ))
        indices.append(position)

    array = np.stack(values).astype(np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    channels = 2 if config["hang_planes"] == "dual" else 1
    expected = (len(indices), 5, config["jung_points_per_cycle"]) if method == "jung" else (
        len(indices), channels, config["hang_image_size"], config["hang_image_size"])
    if array.shape != expected:
        raise ValueError(f"invalid input shape {array.shape}, expected {expected}")
    if not np.isfinite(array).all():
        raise ValueError("non-finite model input")
    return array, indices


def get_model(method, config):
    """实例化 V3 已冻结的论文网络重建；V6 只按平面数调整 Hang 的输入通道。"""
    if method == "jung":
        return JungModifiedCNN(in_channels=5, num_classes=NUM_CLASSES)
    if method == "hang":
        channels = 2 if config["hang_planes"] == "dual" else 1
        return HangWCNN(in_channels=channels, num_classes=NUM_CLASSES,
                        image_size=config["hang_image_size"])
    raise ValueError("unknown model " + str(method))


def image_audit(images, labels):
    """统计轨迹图的退化程度，用于判断 Hang 的输入是否还有判别信息。

    返回：
        distinct：不同图像的个数。
        duplicate_groups：同一图像跨不同标签出现的组数。
        duplicate_samples：这些组涉及的样本数。
        on_unit_circle_fraction：落在半径 1 上的像素占全部点亮像素的比例。
            标量归一化下故障窗应明显小于 1；逐采样时刻归一化下它必然等于 1，
            是 V5 失效最直接的可量化指标。
        on_unit_circle_fraction_by_channel：逐通道的同一比例。双平面输入下
            第 0 通道是基波平面，第 1 通道是三次平面。
    """
    labels = np.asarray(labels)
    hashes = [hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
              for image in images]
    groups = {}
    for digest, label in zip(hashes, labels):
        groups.setdefault(digest, set()).add(int(label))
    duplicate_groups = sum(1 for value in groups.values() if len(value) > 1)
    duplicate_samples = sum(sum(1 for d in hashes if d == digest)
                            for digest, value in groups.items() if len(value) > 1)

    radius_fraction = None
    by_channel = None
    flat = np.asarray(images)
    if flat.ndim == 4 and flat.shape[-2] == flat.shape[-1]:
        size = flat.shape[-1]
        coordinates = np.arange(size)
        # 像素中心到图像中心的归一化距离，用来判断点亮像素是否全在半径 1 附近。
        center = (size - 1) / 2.0
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        radius = np.hypot(xx - center, yy - center) / center
        lit = flat > 0.5
        total = int(lit.sum())
        if total:
            radius_fraction = float((lit & (radius > 0.93)).sum()) / total
        by_channel = []
        for index in range(flat.shape[1]):
            channel_lit = lit[:, index]
            channel_total = int(channel_lit.sum())
            by_channel.append(
                float((channel_lit & (radius > 0.93)).sum()) / channel_total
                if channel_total else None)
    return {
        "distinct": len(groups),
        "duplicate_groups": int(duplicate_groups),
        "duplicate_samples": int(duplicate_samples),
        "on_unit_circle_fraction": radius_fraction,
        "on_unit_circle_fraction_by_channel": by_channel,
    }


@torch.no_grad()
def smoke(config):
    """CPU 前向形状检查，只验证接口，不验证诊断性能。"""
    jung = get_model("jung", config)
    hang = get_model("hang", config)
    channels = 2 if config["hang_planes"] == "dual" else 1
    a = tuple(jung(torch.randn(2, 5, config["jung_points_per_cycle"])).shape)
    b = tuple(hang(torch.randn(2, channels, config["hang_image_size"],
                               config["hang_image_size"])).shape)
    if not a == b == (2, NUM_CLASSES):
        raise AssertionError(f"unexpected output shapes {a} {b}")
    return {"jung": a, "hang": b}
