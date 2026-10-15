"""构造全63文件的仿真Train/Val周期划分，保留旧协议和模型结构。

直接读取作者MAT与Excel。健康片段去重后，各片段按时间分块，边界空出
完整周期；Train/Val覆盖所有实际存在的工况与故障组合。输出包含可移植
的只读原始文件副本、标签来源、窗口清单和核验记录，不读取实测波形。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np

from audit_dataset import FAULT_NAMES, labels_from_xlsx
from prepare import matrix, windows
from v2_experiment import raw_cycles, fixed_noisy_validation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/source_cycle_split_v2.json"
DEFAULT_OUT = ROOT / "outputs/source_cycle_split_v2"
PROFILE = "hust-source-cycle-split-v2"


def read(path: Path) -> dict | list:
    """读取UTF-8元数据；路径由命令行或本协议显式提供。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    """逐块计算输入字节指纹，用于准备、迁移和读取时核对。"""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value) -> None:
    """输出可直接检查的JSON；调用者保证整个输出目录为新目录。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def filename_label(record_id: str) -> dict:
    """解析S-转速-负载-故障编号-O；故障名称沿用作者Excel中的映射。"""
    match = re.fullmatch(r"S-(\d+)-(\d+)-([1-5])-O", record_id)
    if match is None:
        raise ValueError(f"文件名不符合已验证命名规则：{record_id}")
    speed, torque, label = map(int, match.groups())
    return {"speed": speed, "torque": torque, "fault_label": label,
            "fault_name": FAULT_NAMES[label]}


def split_positions(count: int, validation_fraction: float, gap_cycles: int) -> tuple[int, int]:
    """返回训练结束位置、验证开始位置；验证数量按可用周期数四舍五入。

    每个片段独立保留验证周期，因此小片段的实际比例不必精确等于20%。
    中间的gap周期不进入任一集合，保证插值读取也不跨Train/Val边界。
    """
    if not 0 < validation_fraction < 1 or gap_cycles < 1:
        raise ValueError("验证比例须在0与1之间，隔离周期至少为1")
    available = count - gap_cycles
    validation = max(1, int(math.floor(available * validation_fraction + 0.5)))
    training = available - validation
    if training < 1:
        raise ValueError(f"片段只有{count}个周期，无法同时保留Train/Gap/Val")
    return training, training + gap_cycles


def read_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只读取五相仿真I_We，沿用旧协议10kHz时间轴与角速度切周期。"""
    value = matrix(path, "I_We")
    if value.shape != (10000, 6) or np.any(value[:, -1] <= 0):
        raise ValueError(f"仿真矩阵或角速度不符合原协议：{path}")
    return np.arange(10000) / 10000.0, value[:, :5].copy(), value[:, -1].copy()


def make_windows(records: dict, labels: dict, config: dict) -> tuple[dict, list]:
    """先对健康电流片段成组去重，再逐片段按时间划分，故障片段全部保留。

    只按完全相同的原始健康电流字节识别副本，不把相近波形当作重复。
    同一健康副本只取一个确定来源；故障编号不用于决定周期边界。
    """
    families = defaultdict(list)
    for rid, (time, currents, _) in records.items():
        selected = (time >= 0.1) & (time <= 0.45)
        fingerprint = hashlib.sha256(currents[selected].tobytes()).hexdigest()
        families[fingerprint].append(rid)
    healthy_ids = {min(group) for group in families.values()}
    family_rows = [{"sha256": key, "records": sorted(group), "retained": min(group)}
                   for key, group in sorted(families.items())]
    parts = {name: [] for name in ("train", "val", "gap")}
    for rid, (time, currents, omega) in sorted(records.items()):
        candidates = windows(time, omega)
        for state in (0, 1):
            if state == 0 and rid not in healthy_ids:
                continue
            segment = [row for row in candidates if row["state"] == state]
            train_end, val_start = split_positions(len(segment), config["validation_fraction"], config["gap_cycles"])
            label = 0 if state == 0 else labels[rid]["fault_label"]
            for position, row in enumerate(segment):
                part = "train" if position < train_end else ("gap" if position < val_start else "val")
                parts[part].append({"record_id": rid, "label": label, "state": state,
                                    "cycle_index": position, "start": row["start"], "end": row["end"]})
    return parts, family_rows


def verify_split(records: dict, labels: dict, parts: dict, config: dict) -> dict:
    """核查实际覆盖、周期隔离、插值支撑点及完整样本的跨集合完全重复。

    同文件可以同时包含Train和Val，核查单位是片段及其时间范围，不能再
    使用旧协议的“文件编号必须互斥”。本核查不宣称相邻稳态周期统计独立。
    """
    expected = {(row["speed"], row["torque"], row["fault_label"]) for row in labels.values()}
    segments = defaultdict(dict)
    result = {}
    for part, rows in parts.items():
        for row in rows:
            segments[row["record_id"], row["state"]].setdefault(part, []).append(row)
        seen = {(labels[row["record_id"]]["speed"], labels[row["record_id"]]["torque"], row["label"])
                for row in rows if row["state"] == 1}
        if seen != expected or set(row["label"] for row in rows) != set(range(6)):
            raise ValueError(f"{part}未覆盖所有实际存在的工况/类别")
        result[part] = {"samples": len(rows), "records": len({row["record_id"] for row in rows}),
                        "class_counts": [Counter(row["label"] for row in rows)[i] for i in range(6)]}
    minimum_gap = float("inf")
    for (rid, _), group in segments.items():
        train, val, gap = group["train"], group["val"], group["gap"]
        if len(gap) != config["gap_cycles"]:
            raise ValueError("片段隔离周期数量改变")
        last, first = train[-1], val[0]
        separation = first["start"] - last["end"]
        if separation <= 0:
            raise ValueError("Train/Val时间范围重叠")
        time = records[rid][0]
        # 线性插值可能读取边界外相邻原始点；仍必须保留不重叠的支撑范围。
        train_right = np.searchsorted(time, last["end"], side="left")
        val_left = np.searchsorted(time, first["start"], side="right") - 1
        if train_right >= val_left:
            raise ValueError("Train/Val插值支撑点相交")
        minimum_gap = min(minimum_gap, separation)
    fingerprints = {}
    for part in ("train", "val"):
        samples = raw_cycles(records, parts[part], points=config["raw_points"])
        fingerprints[part] = {hashlib.sha256(sample.tobytes()).hexdigest() for sample in samples}
    shared = fingerprints["train"] & fingerprints["val"]
    if shared:
        raise ValueError(f"存在{len(shared)}种跨集合完全重复周期")
    return {"status": "passed", "sets": result, "segments_checked": len(segments),
            "minimum_gap_seconds": minimum_gap, "cross_split_exact_cycle_duplicates": len(shared),
            "all_observed_fault_conditions_in_train_and_val": True,
            "same_record_time_blocks_are_not_independent_acquisitions": True,
            "target_waveforms_read": False, "models_trained": False}


def prepare(source_root: Path, labels_xlsx: Path, out: Path, config_path: Path = CONFIG) -> None:
    """从63个原始文件创建独立数据版本；输出存在时拒绝覆盖。"""
    config = read(config_path)
    if config["protocol"] != PROFILE or config["raw_points"] != 4096 or config["source_fs_hz"] != 10000:
        raise ValueError("数据协议或4096点长度改变")
    if out.exists():
        raise FileExistsError(out)
    files = sorted(source_root.glob("*.mat"))
    if len(files) != config["source_files"]:
        raise ValueError(f"需要63个原始MAT，实际{len(files)}个：{source_root}")
    authored = labels_from_xlsx(labels_xlsx)
    if len(authored) != 59 or not set(authored) <= {p.stem for p in files}:
        raise ValueError("作者Excel标签覆盖与已核查的59文件不一致")
    labels, records, inputs = {}, {}, {}
    for path in files:
        rid = path.stem
        label = filename_label(rid)
        if rid in authored and label != authored[rid]:
            raise ValueError(f"命名与Excel标签不一致：{rid}")
        labels[rid] = {**label, "label_source": "author_xlsx" if rid in authored else "filename_inferred_user_authorized"}
        records[rid] = read_source(path)
        inputs[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    parts, families = make_windows(records, labels, config)
    verification = verify_split(records, labels, parts, config)
    out.mkdir(parents=True)
    (out / "raw_source").mkdir()
    for path in files:
        # 保留可移植的本轮输入副本，不回写用户原始目录，也不依赖本机绝对路径。
        shutil.copyfile(path, out / "raw_source" / path.name)
        if sha(out / "raw_source" / path.name) != inputs[path.name]["sha256"]:
            raise IOError(f"输入副本校验失败：{path.name}")
    save(out / "SOURCE_LABELS.json", labels)
    save(out / "SOURCE_INPUTS.json", inputs)
    save(out / "HEALTHY_DUPLICATE_FAMILIES.json", families)
    for part, rows in parts.items():
        save(out / f"source_{part}_windows.json", rows)
    manifest = {"protocol": PROFILE, "config": config, "source_records": len(records),
                "conditions": [list(group) for group in sorted({(r["speed"], r["torque"]) for r in labels.values()})],
                "label_sources": dict(Counter(row["label_source"] for row in labels.values())),
                "author_xlsx_sha256": sha(labels_xlsx),
                "healthy_unique_segments": len(families), "sets": verification["sets"],
                "source_test_created": False, "target_split_changed": False,
                "metadata_sha256": {p.name: sha(p) for p in sorted(out.glob("*.json"))}}
    save(out / "SOURCE_SPLIT.json", manifest)
    save(out / "VERIFICATION.json", verification)
    print(json.dumps(verification, ensure_ascii=False), flush=True)


def source_provenance(root: Path) -> dict:
    """为训练结果记录数据版本、窗口指纹及来源，避免与旧划分成绩混用。"""
    manifest = read(root / "SOURCE_SPLIT.json")
    if manifest["protocol"] != PROFILE:
        raise ValueError("不是本轮数据划分")
    for name, expected in manifest["metadata_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"数据元信息指纹改变：{name}")
    return {"protocol": PROFILE, "manifest_sha256": sha(root / "SOURCE_SPLIT.json"),
            "sets": manifest["sets"], "source_records": manifest["source_records"],
            "target_split_changed": False}


def source_views(root: Path):
    """提供通用训练读取接口，形状与旧source_views一致，样本数来自新清单。

    Train和Val共享原记录编号，但只读取各自时间块。训练噪声仍由原函数
    每轮重抽；验证噪声按固定rid/SNR种子分别生成，不复用训练噪声。
    """
    provenance = source_provenance(root)
    labels = read(root / "SOURCE_LABELS.json")
    inputs = read(root / "SOURCE_INPUTS.json")
    records = {}
    for name, expected in inputs.items():
        path = root / "raw_source" / name
        if sha(path) != expected["sha256"]:
            raise ValueError(f"原始仿真输入改变：{name}")
        records[Path(name).stem] = read_source(path)
    train, val = (read(root / f"source_{part}_windows.json") for part in ("train", "val"))
    for part, rows in (("train", train), ("val", val)):
        if len(rows) != provenance["sets"][part]["samples"]:
            raise ValueError("样本数量与冻结清单不一致")
        for row in rows:
            expected = 0 if row["state"] == 0 else labels[row["record_id"]]["fault_label"]
            if row["label"] != expected:
                raise ValueError("周期标签与已核对标签字典不一致")
    y = np.asarray([row["label"] for row in train], dtype=np.int64)
    vy = np.asarray([row["label"] for row in val], dtype=np.int64)
    clean = raw_cycles(records, train)
    val_clean = raw_cycles(records, val)
    noise = fixed_noisy_validation(records, val, noise_levels=read(root / "SOURCE_SPLIT.json")["config"]["source_val_noise_db"])
    return (records, train, y, clean), (records, val, vy, val_clean, noise)


def main():
    """生成可迁移数据目录；不训练模型，也不读取任何实测文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--labels-xlsx", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.source_root, args.labels_xlsx, args.out)


if __name__ == "__main__":
    main()
