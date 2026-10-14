"""只读分析固定HUST数据的类别差异、工况波动和域间变化，不训练分类器。

比较45维完整复谐波与25维DC/谐波幅值表示；所有近邻都排除同文件。
近邻标签一致性是有标签数据几何诊断，不是新的正式测试准确率。
不使用Final-Holdout，不调整此前模型、特征配置或已选超参数。
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8")
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
import torch


def read(path):
    """读取已冻结数据与元数据，不从文件编号产生输入特征。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    """记录实际输入指纹，分析前后要求不变。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def views(features):
    """原45维及另一个只读物理描述：每相DC＋1–4阶幅值，共25维。

幅值对周期起始位置的循环平移不敏感，DC保留正负偏置；它不是前轮
分类器使用的表示，也不用于重新拟合或选择任何分类器。
    """
    x = features.astype(np.float64).reshape(-1, 5, 9)
    complex_pairs = x[..., 1:].reshape(-1, 5, 4, 2)
    amplitude = np.sqrt((complex_pairs**2).sum(-1))
    return {"complex45": features.astype(np.float64),
            "dc_amplitude25": np.concatenate((x[..., :1], amplitude), axis=-1).reshape(-1, 25)}


def metadata_arrays(metadata):
    """同文件同类别的周期总权重为1，避免长记录拥有更大统计权重。"""
    labels = np.asarray([row["label"] for row in metadata], dtype=np.int64)
    ids = np.asarray([row["record_id"] for row in metadata])
    sizes = Counter(zip(ids, labels))
    weights = np.asarray([1 / sizes[rid, label] for rid, label in zip(ids, labels)])
    return labels, ids, weights


def class_geometry(features, metadata):
    """描述每类均值/类内均方波动，以及15个类对的间距与波动比。

直接使用已归一化物理尺度，不拟合Z-score，不把比例换算成错误率。
    """
    labels, ids, weights = metadata_arrays(metadata)
    stats, pairs = {}, []
    for label in range(6):
        ix = labels == label
        mean = np.average(features[ix], axis=0, weights=weights[ix])
        spread = float(np.sqrt(np.average(((features[ix] - mean)**2).sum(1), weights=weights[ix])))
        stats[str(label)] = {"cycle_count": int(ix.sum()), "record_count": len(set(ids[ix])),
                             "mean": mean.tolist(), "within_class_rms_distance": spread}
    for a in range(6):
        for b in range(a + 1, 6):
            distance = float(np.linalg.norm(np.asarray(stats[str(a)]["mean"]) - stats[str(b)]["mean"]))
            spread = np.sqrt((stats[str(a)]["within_class_rms_distance"]**2 + stats[str(b)]["within_class_rms_distance"]**2) / 2)
            pairs.append({"class_a": a, "class_b": b, "class_centroid_distance": distance,
                          "pooled_within_class_rms_distance": float(spread),
                          "centroid_to_within_ratio": float(distance / spread) if spread > 0 else None})
    return stats, pairs


def nearest_relation(query, qm, reference, rm, exclude_same_record):
    """按欧氏距离核对最近邻标签；同文件两个状态和相邻周期全部排除。

按文件/类别加权后再对六类平均。Dev内部关系会使用其他Dev标签，
因此仅为描述统计，不是源域模型在独立Test上的性能。
    """
    qy, qid, qw = metadata_arrays(qm)
    ry, rid, _ = metadata_arrays(rm)
    distances = cdist(query, reference, metric="sqeuclidean")
    if exclude_same_record:
        distances[qid[:, None] == rid[None, :]] = np.inf
    assert np.isfinite(distances.min(1)).all()
    index = distances.argmin(1)
    correct = ry[index] == qy
    per_class = [float(np.average(correct[qy == label], weights=qw[qy == label])) for label in range(6)]
    return {"equal_class_equal_record_label_agreement_percent": float(np.mean(per_class) * 100),
            "per_class_agreement_percent": [value * 100 for value in per_class],
            "median_nearest_distance": float(np.median(np.sqrt(distances[np.arange(len(index)), index]))),
            "excluded_same_record": exclude_same_record}


def weighted_quantiles(values, weights):
    """按文件/状态等权输出25/50/75分位，保留周期波动而非先平均降噪。"""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    location = (np.cumsum(weights) - .5 * weights) / weights.sum()
    return np.interp([.25, .5, .75], location, values).tolist()


def physical_distributions(features, metadata):
    """逐类五相DC和RMS分布及事先列出的物理类对排序AUC。

RMS由Parseval公式计算。AUC只描述一个指标的排序可分性，方向取
max(AUC,1-AUC)；不是六类准确率，没有选择阈值或拟合分类器。
    """
    x = features.astype(np.float64).reshape(-1, 5, 9)
    dc = x[..., 0]
    rms = np.sqrt(dc**2 + 2 * (x[..., 1:].reshape(-1, 5, 4, 2)**2).sum((-1, -2)))
    labels, ids, weights = metadata_arrays(metadata)
    distributions, pairs = [], []
    for label in range(6):
        ix = labels == label
        distributions.append({"label": label, "dc_per_phase_q25_median_q75": [weighted_quantiles(dc[ix, phase], weights[ix]) for phase in range(5)],
                              "rms_per_phase_q25_median_q75": [weighted_quantiles(rms[ix, phase], weights[ix]) for phase in range(5)]})
    for a, b, kind, phase in ((0, 3, "rms", 0), (0, 5, "rms", 2), (1, 2, "dc", 0),
                              (1, 4, "dc", 1), (2, 5, "rms", 2), (3, 5, "rms", 2)):
        ix = (labels == a) | (labels == b)
        values = dc[ix, phase] if kind == "dc" else rms[ix, phase]
        auc = roc_auc_score(labels[ix] == b, values, sample_weight=weights[ix])
        pairs.append({"class_a": a, "class_b": b, "feature": kind, "phase": "ABCDE"[phase],
                      "direction_free_pairwise_ranking_auc": float(max(auc, 1 - auc))})
    return {"per_class": distributions, "prespecified_physical_pair_auc": pairs}


def main(args):
    """仅读既有Source缓存和88条Dev；输出统计，不加载分类器权重。"""
    started = time.perf_counter()
    assert not args.output.exists()
    sys.path.insert(0, str(args.difficulty_experiment / "code/scripts"))
    sys.path.insert(0, str(args.reference_experiment / "code/scripts"))
    import hust_arch_suite_experiment as exp
    from lp4_difficulty_models import extract_features
    from prepare import OUT as PREPARE_OUT
    exp.fix_randomness(42)
    cache = args.difficulty_experiment / "results/cache/seed42/source_features.npz"
    train_meta = read(PREPARE_OUT / "source_train_windows.json")
    val_meta = read(PREPARE_OUT / "source_val_windows.json")
    frozen = read(PREPARE_OUT / "target_dev_clean_windows.json")
    assert len(train_meta) == 738 and len(val_meta) == 302 and len(frozen) == 2474
    assert not {r["record_id"] for r in train_meta} & {r["record_id"] for r in val_meta}
    protected = {str(path): sha(path) for path in (cache, PREPARE_OUT / "source_train_windows.json",
                                                 PREPARE_OUT / "source_val_windows.json", PREPARE_OUT / "target_dev_clean_windows.json")}
    with np.load(cache, allow_pickle=False) as db:
        datasets = {"source_train_clean": (db["clean"].copy(), train_meta), "source_val_clean": (db["val_clean"].copy(), val_meta)}
        assert np.array_equal(db["labels"], [r["label"] for r in train_meta])
        assert np.array_equal(db["val_labels"], [r["label"] for r in val_meta])
    geometry, relations, physical = {}, [], {}
    train_views = views(datasets["source_train_clean"][0])
    deploy = read(args.reference_experiment / "DEPLOYMENT_MANIFEST.json")
    for level in (None, 10, 5, 0, -5):
        name = "target_dev_clean" if level is None else f"target_dev_snr{level}"
        raw, metadata = (exp.data.target_dev_from_mat(-5, args.minus5_root / "generated/target") if level == -5 else
                         exp.target_raw(level, snr_root=Path(deploy["target_snr_root"]), strong_root=args.strong_root, dev_export_root=None))
        assert metadata == frozen and len(raw) == 88
        features, _ = extract_features(exp.data.raw_cycles(raw, metadata), torch.device("cuda"), verify_reconstruction=True)
        datasets[name] = features, metadata
        print(f"FEATURES_READY: {name}, 88 families/2474 cycles", flush=True)
    for name, (features, metadata) in datasets.items():
        physical[name] = physical_distributions(features, metadata)
        for view_name, matrix in views(features).items():
            stats, pairs = class_geometry(matrix, metadata)
            geometry[name + "/" + view_name] = {"class_statistics": stats, "pair_statistics": pairs}
            within = nearest_relation(matrix, metadata, matrix, metadata, True)
            relations.append({"dataset": name, "representation": view_name, "relation": "within_domain_other_records", **within})
            if name != "source_train_clean":
                across = nearest_relation(matrix, metadata, train_views[view_name], train_meta, False)
                relations.append({"dataset": name, "representation": view_name, "relation": "nearest_source_train", **across})
        print(f"GEOMETRY_COMPLETE: {name}", flush=True)
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    args.output.mkdir(parents=True)
    exp.write_json(args.output / "DATA_GEOMETRY.json", {"status": "passed", "geometry": geometry, "nearest_relations": relations,
                   "physical_distributions": physical, "input_files_sha256": protected, "classifiers_trained": 0,
                   "classifier_weights_loaded": False, "previous_parameters_or_selection_modified": False,
                   "source_test_or_final_holdout_loaded": False, "per_cycle_values_kept_without_record_averaging": True,
                   "within_domain_nearest_statistics_not_formal_test_accuracy": True,
                   "record_label_equal_weights_used": True, "elapsed_seconds": time.perf_counter() - started,
                   "script_sha256": sha(Path(__file__))})
    exp.write_csv(args.output / "NEAREST_RELATIONS.csv", [{k: v for k, v in row.items() if k != "per_class_agreement_percent"} for row in relations])
    exp.write_csv(args.output / "PHYSICAL_PAIR_AUC.csv", [{"dataset": name, **row} for name, obj in physical.items() for row in obj["prespecified_physical_pair_auc"]])
    exp.write_csv(args.output / "PAIR_SEPARATION.csv", [{"dataset_representation": name, **row} for name, obj in geometry.items() for row in obj["pair_statistics"]])
    print("ALL_COMPLETE: data geometry only, no classifier fitting", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-experiment", "difficulty-experiment", "strong-root", "minus5-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
