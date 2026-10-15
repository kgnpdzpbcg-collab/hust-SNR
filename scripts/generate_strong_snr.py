"""生成HUST仿真与实测5/0或单独−5/−10 dB派生MAT，保留原列结构、编号和划分。

仿真：每相按整记录RMS校准高斯噪声，验证集沿用原固定随机数规则。
实测：从作者干净/10 dB配对提取噪声，放大到指定强噪声档位；不假定
作者噪声是高斯分布，不给已经含噪的波形盲目叠加第二层噪声。
本脚本只生成和核验数据，不训练、不选择或评估模型。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def read(path):
    """读取生成配置、标签和冻结划分，错误直接报告。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    """分块核对原输入和生成文件，不把大型MAT整体留在内存。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load_matrix(path, field, shape):
    """只接受原审计确认的字段、float64维度和有限值。"""
    value = loadmat(path, variable_names=[field])[field]
    if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError(f"MAT字段、维度或数值异常：{path}")
    return value


def phase_snr(clean, noisy):
    """逐相计算整记录SNR；不是每个周期或仅故障段的SNR。"""
    signal = np.mean(clean ** 2, axis=0)
    noise = np.mean((noisy - clean) ** 2, axis=0)
    if np.any(signal <= 0) or np.any(noise <= 0):
        raise ValueError("整记录信号或噪声功率为零，无法定义SNR")
    return 10 * np.log10(signal / noise)


def gaussian_current(clean, snr_db, seed):
    """按原仿真增强公式生成噪声，输入/输出均为[时间点,5相]电流。

    用真实噪声样本的均方功率校准，使每相整记录SNR精确等于指定档位。
    不扣除噪声样本均值，保持与原calibrated_noise处理一致。
    """
    # scipy读入MAT通常是列连续数组；旧record()先copy成行连续数组。
    # 统一内存顺序，避免mean在不同排列下出现末位舍入差异，使固定
    # 源验证噪声不仅SNR一致，也与旧calibrated_noise逐数值一致。
    clean = np.ascontiguousarray(clean, dtype=np.float64)
    noise = np.random.default_rng(seed).standard_normal(clean.shape)
    power = np.mean(clean.astype(np.float64) ** 2, axis=0)
    noise *= np.sqrt(power / np.mean(noise ** 2, axis=0)) / (10 ** (snr_db / 20))
    return clean + noise


def scaled_author_current(clean, author_snr10, snr_db):
    """从作者10 dB版本提取噪声再缩放，保持原时间/幅度变化和频谱形态。

    各档复用同一个作者噪声样本，因此不能当成独立采集或独立随机噪声。
    5/0/−5/−10 dB分别乘10**0.25/0.5/0.75/1；返回新的电流，不改输入数组。
    """
    error = np.max(np.abs(phase_snr(clean, author_snr10) - 10))
    if error > 1e-6:
        raise ValueError(f"作者10 dB参考文件核算不符：最大差{error}")
    return clean + (author_snr10 - clean) * (10 ** ((10 - snr_db) / 20))


def source_seed(partition, rid, snr, base_seed):
    """训练/保留集使用独立命名随机流；验证保留原固定Val随机流格式。"""
    key = (f"HUST-V1:val:{snr}:{rid}" if partition == "val" else
           f"HUST-STRONG-v1:{partition}:{snr}:{rid}:{base_seed}")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def save_checked(path, field, value):
    """写入新目录后重新加载核对，记录文件及矩阵指纹。

    MAT文件头包含导出时间；矩阵指纹用于核对固定随机种子的数值复现，
    文件指纹用于核对本次实际导出文件的完整字节。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {field: value}, do_compression=True)
    restored = loadmat(path, variable_names=[field])[field]
    if not np.array_equal(restored, value):
        raise RuntimeError(f"MAT重新加载与生成矩阵不一致：{path}")
    return {"bytes": path.stat().st_size, "file_sha256": sha(path),
            "matrix_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def generate(args):
    """按配置流式生成每档59个仿真和440个实测文件，不覆盖旧文件。"""
    config, labels, split, inputs = map(read, [args.config, args.labels, args.split, args.input_manifest])
    levels = config["generated_snr_db"]
    # 保留原5/0协议；−5/−10各使用独立配置和新目录，不修改旧实验档位。
    if levels not in ([5, 0], [-5], [-10]) or args.output.exists():
        raise ValueError("本协议只接受[5,0]、[-5]或[-10]，输出目录必须不存在")
    source_part = {rid: part for part, ids in split["source_records"].items() for rid in ids}
    if set(source_part) != set(labels["simulation"]):
        raise ValueError("仿真标签与原训练/验证/保留划分不一致")
    if len(source_part) != 59 or len(labels["experimental"]) != 440:
        raise ValueError("预期59个已标注仿真及440个已标注实测编号")
    source_count = len(source_part) * len(levels)
    target_count = len(labels["experimental"]) * len(levels)
    # 先核对全部原输入的传输指纹，避免写了一部分才发现文件传错。
    for rid in source_part:
        path = args.source_root / f"{rid}.mat"
        if sha(path) != inputs["source_clean_sha256"][path.name]:
            raise ValueError(f"仿真原文件指纹不一致：{rid}")
    for key, entry in inputs["target_files"].items():
        folder, filename = key.split("/")
        path = (args.target_clean_root if folder == "target_clean" else args.target_snr10_root) / filename
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"实测原文件指纹不一致：{key}")
    args.output.mkdir(parents=True)
    records = []
    dev, holdout = set(split["target_dev_records"]), set(split["target_holdout_records"])
    canonical = {rid: rid for rid in labels["experimental"]}
    for group in split["target_duplicate_groups"]:
        canonical.update({rid: min(group) for rid in group})
    for rid in sorted(source_part):
        clean = load_matrix(args.source_root / f"{rid}.mat", "I_We", (10000, 6))
        for snr in levels:
            seed = source_seed(source_part[rid], rid, snr, config["base_seed"])
            value = clean.copy()
            value[:, :5] = gaussian_current(clean[:, :5], snr, seed)
            measured = phase_snr(clean[:, :5], value[:, :5])
            assert np.max(np.abs(measured - snr)) < 1e-6
            assert np.array_equal(clean[:, -1], value[:, -1])
            name = f"source/SNR_{snr}/{rid}.mat"
            records.append({"file": name, "domain": "simulation", "record_id": rid,
                            "partition": source_part[rid], "snr_db": snr, "rng_seed": seed,
                            "measured_phase_snr_db": measured.tolist(),
                            **save_checked(args.output / name, "I_We", value)})
    print(f"SOURCE_COMPLETE: {source_count} MAT", flush=True)
    for pos, rid in enumerate(sorted(labels["experimental"]), 1):
        clean = load_matrix(args.target_clean_root / f"{rid}.mat", "name", (10031, 7))
        author = load_matrix(args.target_snr10_root / f"{rid}.mat", "I_noisy", (10031, 7))
        assert np.array_equal(clean[:, 0], author[:, 0]) and np.array_equal(clean[:, -1], author[:, -1])
        assert np.all(np.diff(clean[:, 0]) >= 0)
        partition = "dev" if rid in dev else "final_holdout" if rid in holdout else "duplicate_alias"
        for snr in levels:
            value = clean.copy()
            value[:, 1:6] = scaled_author_current(clean[:, 1:6], author[:, 1:6], snr)
            measured = phase_snr(clean[:, 1:6], value[:, 1:6])
            assert np.max(np.abs(measured - snr)) < 1e-6
            assert np.array_equal(clean[:, [0, -1]], value[:, [0, -1]])
            name = f"target/SNR_{snr}/{rid}.mat"
            records.append({"file": name, "domain": "experimental", "record_id": rid,
                            "partition": partition, "canonical_record_id": canonical[rid],
                            "snr_db": snr, "noise_multiplier_from_author10": 10 ** ((10 - snr) / 20),
                            "measured_phase_snr_db": measured.tolist(),
                            **save_checked(args.output / name, "I_noisy", value)})
        if pos % 100 == 0 or pos == 440:
            print(f"TARGET_PROGRESS: {pos}/440 records", flush=True)
    assert len(records) == source_count + target_count
    manifest = {"status": "passed", "config": config, "files": records,
                "file_count": len(records), "source_files": source_count, "target_files": target_count,
                "max_original_record_phase_snr_error_db": max(
                    abs(v - row["snr_db"]) for row in records for v in row["measured_phase_snr_db"]),
                "config_sha256": sha(args.config), "labels_sha256": sha(args.labels),
                "split_sha256": sha(args.split), "input_manifest_sha256": sha(args.input_manifest),
                "generator_sha256": sha(Path(__file__)), "models_trained_or_evaluated": False,
                "old_files_overwritten": False, "target_final_holdout_generated_but_not_evaluated": True}
    (args.output / "GENERATION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "config")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source-root", "target-clean-root", "target-snr10-root", "labels",
                   "split", "config", "input-manifest", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    generate(parser.parse_args())
