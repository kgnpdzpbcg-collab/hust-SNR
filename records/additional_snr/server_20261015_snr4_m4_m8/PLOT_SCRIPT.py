"""新周期划分下四模型的分类前128维t-SNE；仅读取42/44的最佳权重。

提取、投影和绘图分阶段保存，不训练或重选模型，不覆盖六种子正式结果。
所有模型都完整前向，通过分类器输入钩子取特征，避免绕过B1/LP4前端。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time


def read(path):
    """读取明确给定的JSON；数据、权重和历史记录不在本入口改写。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    """把可追溯记录另存到本轮目录，禁止修改被保护的旧文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    """用文件字节SHA验证代码、权重、预测和导出的图像。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    """保存未舍入的指标和坐标；标签只用于指标和着色，不用于降维拟合。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup(args):
    """导入原训练代码快照，确保前端、模型和数据读取与正式评估一致。"""
    sys.path.insert(0, str(args.training_root / "code/scripts"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(args.config)
    assert config["models"] == ["resnet_lp4", "cnn1d_plain", "resnet1d", "b1"]
    assert config["seeds"] == [42, 44]
    # 新档位使用独立配置；旧配置和历史输出保持原样，不接受任意挑选条件。
    supported = {
        "source_cycle_v2_four_models_two_display_seeds_tsne_v1":
            ["clean", "snr5", "snr0", "snr-5", "snr-10"],
        "source_cycle_v2_additional_snr4_minus4_minus8_tsne_v1":
            ["snr4", "snr-4", "snr-8"],
    }
    assert config["conditions"] == supported[config["protocol"]]
    if config["protocol"] == "source_cycle_v2_additional_snr4_minus4_minus8_tsne_v1":
        assert args.snr_root is not None and args.prediction_root is not None
    assert config["feature_dimensions"] == 128
    deployment = read(args.training_root / "DEPLOYMENT_MANIFEST.json")
    assert deployment["code_commit"] == config["source_training_commit"]
    # 逐个核对原快照中的全部代码/配置，不能借用工作区后来修改的模型。
    for name, expected in deployment["code_sha256"].items():
        assert sha(args.training_root / "code" / name) == expected, name
    return config, deployment


def extract(args):
    """钩取8份best的Linear输入，并检查各档预测及模型状态保持原样。

    每组输出[2474,128]。B1没有return_features接口，钩子直接观察原
    backbone.classifier的输入；其他三模型同样取分类器输入以统一位置。
    """
    import numpy as np
    import torch
    config, deployment = setup(args)
    import source_cycle_benchmark as original
    from v1_noise_consistency import fix_randomness
    if args.prediction_root is not None:
        assert read(args.prediction_root / "VERIFICATION.json")["status"] == "passed"
    assert not args.output.exists(), "拒绝覆盖已有分析目录"
    args.output.mkdir(parents=True)
    assert read(args.training_root / "results/reports/VERIFICATION.json")["status"] == "passed"
    protected = {}
    for name in ("PER_SEED_RESULTS.csv", "SUMMARY.csv", "VERIFICATION.json"):
        path = args.training_root / "results/reports" / name
        protected[str(path)] = sha(path)
    models = {}
    for method in config["models"]:
        for seed in config["seeds"]:
            folder = args.training_root / f"results/models/{method}/seed{seed}"
            for name in ("best.pt", "last.pt", "run.json", "history.json"):
                protected[str(folder / name)] = sha(folder / name)
            saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            assert (saved["method"], saved["seed"], saved["protocol"]) == (method, seed, original.PROTOCOL)
            assert saved["source_split"]["manifest_sha256"] == config["source_split_manifest_sha256"]
            assert not saved["final_holdout_evaluated"]
            fix_randomness(seed)
            model = original.build_model(method).cuda().eval()
            model.load_state_dict(saved["state"], strict=True)
            head = model.classifier if method in ("cnn1d_plain", "resnet1d") else model.backbone.classifier
            assert isinstance(head, torch.nn.Linear) and head.in_features == 128 and head.out_features == 6
            models[method, seed] = (model, head, saved["selected"])
    entries, common_metadata = [], None
    for condition in config["conditions"]:
        level = None if condition == "clean" else int(condition[3:])
        snr_root = args.snr_root if args.snr_root is not None else Path(deployment["snr_root"])
        x, metadata = original.target_cycles(level, snr_root,
            Path(deployment["strong_root"]), Path(deployment["extreme_root"]))
        if common_metadata is None:
            common_metadata = metadata
        assert metadata == common_metadata
        truth = np.asarray([row["label"] for row in metadata], dtype=np.int64)
        ids = np.asarray([row["record_id"] for row in metadata])
        assert len(x) == 2474 and len(set(ids)) == 88
        for (method, seed), (model, head, selected) in models.items():
            prediction_root = args.prediction_root if args.prediction_root is not None else args.training_root / "results"
            source = prediction_root / f"evaluations/{method}/seed{seed}/target_dev_{condition}.json"
            protected[str(source)] = sha(source)
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            parts, logits = [], []

            def capture(_module, inputs):
                """只读取原Linear收到的向量；不更改输入、输出、前端或BN状态。"""
                parts.append(inputs[0].detach().cpu().numpy().copy())

            hook = head.register_forward_pre_hook(capture)
            try:
                with torch.inference_mode():
                    for start in range(0, len(x), 128):
                        logits.append(model(torch.from_numpy(x[start:start + 128]).cuda()).cpu().numpy())
            finally:
                hook.remove()
            features, logits = np.concatenate(parts), np.concatenate(logits)
            assert features.shape == (2474, 128) and np.isfinite(features).all()
            predicted = logits.argmax(1)
            prior = read(source)
            assert np.array_equal(truth, prior["truth"]) and np.array_equal(predicted, prior["predicted"])
            # 用钩取向量再走原分类头，核对logits及预测，而非仅确认数组形状。
            with torch.inference_mode():
                replay = np.concatenate([head(torch.from_numpy(features[i:i + 128]).cuda()).cpu().numpy()
                                         for i in range(0, len(features), 128)])
            assert np.array_equal(logits, replay)
            assert all(torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items())
            path = args.output / f"features/{method}_seed{seed}_{condition}_classifier_input128.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, features=features, truth=truth, predicted=predicted, record_ids=ids)
            entries.append({"model": method, "seed": seed, "condition": condition,
                "layer": config["feature_layer"], "feature_dim": 128, "file": str(path),
                "feature_file_sha256": sha(path), "macro_f1": prior["metrics"]["macro_f1"],
                "selected_epoch": selected["epoch"],
                "weights": str(args.training_root / f"results/models/{method}/seed{seed}/best.pt"),
                "original_prediction_file": str(source), "captured_head_logits_exact": True})
            entries[-1]["weights_sha256"] = protected[entries[-1]["weights"]]
            print("EXTRACTED", method, seed, condition, "128D, logits exact", flush=True)
    expected = len(config["models"]) * len(config["seeds"]) * len(config["conditions"])
    assert len(entries) == expected and all(sha(p) == value for p, value in protected.items())
    write(args.output / "PROTECTED_FILES_SHA256.json", protected)
    write(args.output / "FEATURE_MANIFEST.json", {"protocol": config, "files": entries,
        "class_counts": np.bincount(truth, minlength=6).tolist(), "original_states_unchanged": True,
        "model_retraining": False, "all_six_seed_results_preserved": True})


def project(args):
    """复用已核对的旧投影函数，各档均采用相同PCA50和t-SNE参数。

    四CPU进程并发，每进程四线程；每组独立拟合。全局RMS仅乘一个标量，
    不做逐样本或逐维标准化；原128维轮廓系数和跨记录5NN一并记录。
    """
    config, _ = setup(args)
    from six_model_tsne import project_one
    assert not (args.output / "coordinates").exists()
    manifest = read(args.output / "FEATURE_MANIFEST.json")
    expected = len(config["models"]) * len(config["seeds"]) * len(config["conditions"])
    assert manifest["protocol"] == config and len(manifest["files"]) == expected
    started, results = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(project_one, (entry, str(args.output), config)) for entry in manifest["files"]]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print("PROJECTED", len(results), f"/{expected}", row["model"], row["seed"], row["condition"], flush=True)
    results.sort(key=lambda r: (config["conditions"].index(r["condition"]), r["seed"], config["models"].index(r["model"])))
    fields = ("model", "seed", "condition", "feature_dim", "silhouette_macro", "cross_record_5nn_purity_macro",
              "tsne_trustworthiness_k15", "classifier_macro_f1_percent", "pca50_explained_variance_ratio_sum")
    save_csv(args.output / "FEATURE_SEPARABILITY.csv", [{k: r[k] for k in fields} for r in results])
    assert all(sha(p) == value for p, value in read(args.output / "PROTECTED_FILES_SHA256.json").items())
    write(args.output / "SUMMARY.json", {"protocol": config, "results": results,
        "projection_wall_seconds": time.monotonic() - started, "model_retraining": False,
        "independent_tsne_fits": True, "coordinates_comparable_across_panels": False})


def plot(args):
    """每档一张两行四列图：行固定种子，列固定模型；同类始终同色同形。

    每个面板使用所有2474周期，按固定无标签随机顺序画点以减小遮挡顺序偏差。
    各面板保持正方形、独立坐标，保留所有点；图例单独放在底部，不覆盖数据。
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    summary = read(args.output / "SUMMARY.json")
    config = summary["protocol"]
    assert config == read(args.config)
    folder = args.output / "figures"
    assert not folder.exists()
    folder.mkdir()
    names = {"resnet_lp4": "ResNet + LP4", "cnn1d_plain": "Plain CNN1D", "resnet1d": "ResNet1D", "b1": "B1"}
    labels = ["Healthy", "OUS(A)", "OLS(A)", "OP(A)", "OUS(A) & OUS(B)", "OLS(A) & OP(C)"]
    colors = ["#767676", "#0F4D92", "#B64342", "#8BCF8B", "#42949E", "#9A4D8E"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "text.usetex": False})
    files, layouts = [], []
    order = np.random.default_rng(20261015).permutation(2474)
    reference = None
    for condition in config["conditions"]:
        fig, axes = plt.subplots(2, 4, figsize=(180 / 25.4, 112 / 25.4))
        fig.subplots_adjust(left=.064, right=.989, top=.85, bottom=.20, wspace=.13, hspace=.38)
        for r, seed in enumerate(config["seeds"]):
            for c, method in enumerate(config["models"]):
                ax = axes[r, c]
                entry = next(x for x in summary["results"] if (x["model"], x["seed"], x["condition"]) == (method, seed, condition))
                source = args.output / "coordinates" / Path(entry["coordinate_file"]).name
                assert sha(source) == entry["coordinate_file_sha256"]
                with source.open(encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 2474 and [int(x["index"]) for x in rows] == list(range(2474))
                truth = np.asarray([int(x["truth"]) for x in rows])
                if reference is None:
                    reference = truth
                assert np.array_equal(truth, reference)
                coords = np.asarray([[float(x["tsne_x"]), float(x["tsne_y"])] for x in rows])
                # 逐点统一随机顺序；使用同形圆点，颜色编码真实类别，保持遮挡处理一致。
                ax.scatter(coords[order, 0], coords[order, 1], c=np.asarray(colors)[truth[order]],
                    s=3.2, alpha=.68, edgecolors="none", linewidths=0)
                center = (coords.min(0) + coords.max(0)) / 2
                radius = np.ptp(coords, axis=0).max() * .55
                ax.set_xlim(center[0] - radius, center[0] + radius)
                ax.set_ylim(center[1] - radius, center[1] + radius)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_title(f"{names[method]}\nMacro-F1 {entry['classifier_macro_f1_percent']:.2f}%", pad=3)
                if c == 0:
                    ax.set_ylabel(f"Seed {seed}", fontsize=8, labelpad=7)
        condition_name = "Clean" if condition == "clean" else condition[3:] + " dB"
        fig.suptitle(f"Classifier-input features (128D)  |  {condition_name}", y=.975, fontsize=10)
        handles = [Line2D([], [], marker="o", color=colors[i], linestyle="none", markersize=3.8, label=labels[i]) for i in range(6)]
        legend = fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.52, .057),
            fontsize=6.6, frameon=False, handletextpad=.35, columnspacing=1.6, labelspacing=.55)
        note = fig.text(.52, .025, "Selected display seeds 42/44; all 2,474 Dev cycles; true labels; separate t-SNE fits.",
            ha="center", va="center", fontsize=6.3, color="#444444")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        rectangles = [ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()) for ax in axes.flat]
        assert all(abs(box.width - box.height) * 72 < 1.5 for box in rectangles)
        assert np.ptp([box.width for box in rectangles]) * 72 < 1.5
        assert legend.get_window_extent(renderer).y1 < min(axes[1, c].get_window_extent(renderer).y0 for c in range(4))
        assert note.get_window_extent(renderer).y1 < legend.get_window_extent(renderer).y0
        for extension in ("png", "pdf"):
            path = folder / f"tsne_{condition}_seeds42_44.{extension}"
            fig.savefig(path, dpi=600, facecolor="white", transparent=False)
            files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
        layouts.append({"condition": condition, "panels": 8, "all_cycles_drawn": True,
            "square_aligned_panels": True, "legend_note_overlap": False})
        plt.close(fig)
        print("FIGURE_EXPORTED", condition, flush=True)
    write(folder / "FIGURE_MANIFEST.json", {"figures": len(config["conditions"]),
        "panels": len(config["conditions"]) * len(config["seeds"]) * len(config["models"]), "png_dpi": 600,
        "width_mm": 180, "height_mm": 112, "files": files, "layout_checks": layouts,
        "font": "DejaVu Sans", "colors": colors, "markers": "all circles", "class_labels": labels,
        "plot_script_sha256": sha(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__, "python_version": sys.version})


def main():
    """三个明确阶段分别运行，失败时保留现场，不自动更换参数或挑选投影。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "project", "plot"))
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr-root", type=Path, help="新档位MAT父目录；省略时仍读取旧数据")
    parser.add_argument("--prediction-root", type=Path, help="新档位固定权重评价目录；省略时读取原训练结果")
    args = parser.parse_args()
    {"extract": extract, "project": project, "plot": plot}[args.stage](args)


if __name__ == "__main__":
    main()
