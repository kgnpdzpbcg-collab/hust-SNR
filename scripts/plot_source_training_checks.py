"""简单检查A0、B0及LP4＋ResNet原正式训练日志，不训练、不读取实测波形。

输出两张PNG：三模型训练loss/五档源验证F1，以及B0额外预训练与loss分项。
读取原42/43/44全部种子，标记原源验证选中轮次；不平滑、不补造验证loss。
同时保存450行分类日志、30行预训练日志与来源SHA，方便逐点核对。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
COLORS = {42: "#2864B4", 43: "#CB8426", 44: "#668744"}
STYLES = {42: "-", 43: "--", 44: ":"}
LABELS = {"a0": "A0", "b0": "B0", "resnet_lp4": "LP4＋ResNet"}


def read(path):
    """读取原记录，不能依据实测成绩筛选种子。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    """来源与图像使用完整文件SHA指纹。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    """保留未舍入数值，列名说明loss分项与验证F1单位。"""
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decorate(ax, ylabel, log=False, maximum_epoch=50):
    """使用简单白底坐标；loss对数轴显式标注，保留原始逐轮抖动。"""
    ax.set_xlabel("训练轮次")
    ax.set_ylabel(ylabel)
    ax.set_xlim(.5, maximum_epoch + .5)
    ax.set_xticks([1, 10, 20, 30, 40, 50] if maximum_epoch == 50 else [1, 2, 4, 6, 8, 10])
    if log:
        ax.set_yscale("log")
    ax.grid(axis="both", color="#DDDDDD", alpha=.7, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)


def main(output):
    """核对450轮及30轮预训练后，绘制原曲线与源选中轮次。"""
    assert not output.exists(), "检查图使用新目录，不能覆盖已有结果"
    # 本地Windows已安装微软雅黑；找不到时直接报错，避免悄悄生成乱码图片。
    font_manager.findfont("Microsoft YaHei", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.unicode_minus": False, "figure.facecolor": "white"})
    records, sources, rows, warmups, selections = {}, {}, [], [], []
    for method in LABELS:
        root = (ROOT / "records/prototypes_a0_b0/server_c0b30c9/results/models" if method != "resnet_lp4"
                else ROOT / "records/architecture_suite/server_5c75bea/results/models")
        for seed in SEEDS:
            folder = root / method / f"seed{seed}"
            history, run = read(folder / "history.json"), read(folder / "run.json")
            assert len(history) == 50 and [r["epoch"] for r in history] == list(range(1, 51))
            assert run["selected"] == history[run["selected"]["epoch"] - 1]
            for path in (folder / "history.json", folder / "run.json"):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
            records[method, seed] = (history, run)
            selections.append({"method": method, "seed": seed, "source_selected_epoch": run["selected"]["epoch"]})
            for entry in history:
                total = entry["train_loss_mean"]
                ce = entry.get("classification_ce_mean", total)
                recovery = entry.get("source_recovery_loss_mean", 0.)
                assert np.isfinite(total) and total > 0
                np.testing.assert_allclose(total, ce + .2 * recovery, atol=1e-6, rtol=0)
                rows.append({"method": method, "seed": seed, "epoch": entry["epoch"],
                             "train_total_loss": total, "classification_ce_loss": ce,
                             "raw_recovery_loss": recovery, "weighted_recovery_loss": .2 * recovery,
                             "source_clean_macro_f1_percent": entry["clean_f1"] * 100,
                             "source_five_noise_mean_macro_f1_percent": entry["noise_mean_f1"] * 100,
                             "source_selected_epoch": run["selected"]["epoch"]})
            if method == "b0":
                assert len(run["pretrain_recovery_losses"]) == 10
                warmups.extend({"seed": seed, "recovery_pretrain_epoch": epoch, "recovery_loss": loss}
                               for epoch, loss in enumerate(run["pretrain_recovery_losses"], 1))
    assert len(rows) == 450 and len(warmups) == 30
    output.mkdir(parents=True)
    write_csv(output / "TRAINING_VALUES.csv", rows)
    write_csv(output / "B0_PRETRAIN_VALUES.csv", warmups)
    write_csv(output / "SOURCE_SELECTED_EPOCHS.csv", selections)

    # 主图：同一列为同一模型，上方训练loss，下方五档含噪源验证F1。
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.0))
    for col, method in enumerate(LABELS):
        for seed in SEEDS:
            history, run = records[method, seed]
            epochs = np.arange(1, 51)
            selected = run["selected"]["epoch"]
            for row, key, multiplier in ((0, "train_loss_mean", 1), (1, "noise_mean_f1", 100)):
                values = [entry[key] * multiplier for entry in history]
                axes[row, col].plot(epochs, values, color=COLORS[seed], linestyle=STYLES[seed], linewidth=1.4)
                axes[row, col].plot(selected, values[selected - 1], marker="^", color=COLORS[seed], markersize=7, linestyle="none")
        decorate(axes[0, col], "训练loss（对数轴）", log=True)
        decorate(axes[1, col], "五档源验证Macro-F1（%）")
        axes[1, col].set_ylim(0, 103)
        axes[0, col].set_title(LABELS[method] + ("：联合训练总loss" if method == "b0" else "：训练loss"))
        axes[1, col].set_title(LABELS[method] + "：0/5/10/20/30 dB源验证均值")
        chosen = "，".join(f"{seed}→{records[method, seed][1]['selected']['epoch']}" for seed in SEEDS)
        axes[0, col].text(.97, .97, "选中轮次\n" + chosen, transform=axes[0, col].transAxes,
                          ha="right", va="top", fontsize=9,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    axes[1, 0].text(.04, .12, "42：第2–50轮均为100%\n43/44：第1–50轮均为100%",
                    transform=axes[1, 0].transAxes, fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[s], linestyle=STYLES[s], label=f"种子{s}") for s in SEEDS]
    handles.append(Line2D([0], [0], color="#555555", marker="^", linestyle="none", label="原源验证选中权重"))
    fig.suptitle("真实域评估前：仿真训练loss与源验证分数（原正式日志）", fontsize=15, y=.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95), ncol=4, frameon=False)
    fig.text(.5, .015, "全部42/43/44种子；每条曲线50轮，不做平滑。B0额外10轮恢复预训练另图展示。没有逐轮验证loss日志。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .89), h_pad=2.0, w_pad=1.6)
    fig.savefig(output / "TRAIN_LOSS_AND_SOURCE_VAL.png", dpi=140)
    plt.close(fig)

    # B0补图：恢复预训练按原10轮单列，分类阶段的CE与恢复项分开显示。
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for seed in SEEDS:
        values = records["b0", seed][1]["pretrain_recovery_losses"]
        axes[0].plot(np.arange(1, 11), values, color=COLORS[seed], linestyle=STYLES[seed], marker="o", markersize=3, label=f"种子{seed}")
    decorate(axes[0], "恢复loss", maximum_epoch=10)
    axes[0].set_title("B0阶段一：只训练恢复器，固定10轮")
    axes[0].set_ylim(bottom=0)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(frameon=False, ncol=3, fontsize=9, loc="lower left")
    for key, weight, color, style, label in (
        ("classification_ce_mean", 1., "#2864B4", "-", "分类CE"),
        ("source_recovery_loss_mean", .2, "#CB8426", "-", "0.2×恢复loss"),
        ("train_loss_mean", 1., "#666666", "--", "总loss"),
    ):
        values = np.asarray([[row[key] * weight for row in records["b0", seed][0]] for seed in SEEDS])
        axes[1].plot(np.arange(1, 51), values.mean(0), color=color, linestyle=style, linewidth=1.6, label=label)
        axes[1].fill_between(np.arange(1, 51), values.min(0), values.max(0), color=color, alpha=.10)
    decorate(axes[1], "loss（对数轴）", log=True)
    axes[1].set_title("B0阶段二：50轮联合训练，三种子均值")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.text(.5, .02, "右图阴影为三种子最小–最大范围，不是置信区间。loss数值比例不能直接代表梯度贡献。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1), w_pad=2)
    fig.savefig(output / "B0_PRETRAIN_AND_LOSS_COMPONENTS.png", dpi=140)
    plt.close(fig)
    contract = {"purpose": "exploratory training log inspection, not publication figure",
                "renderer": "matplotlib_static_png", "classification_epoch_rows": 450,
                "pretraining_epoch_rows": 30, "methods": list(LABELS), "seeds": list(SEEDS),
                "training_loss_smoothing": False, "train_loss_aggregation": "unweighted mean of 24 batch losses; last batch has 2 source cycles",
                "validation_loss_per_epoch_available": False, "model_weights_or_official_scores_changed": False,
                "source_files_sha256": sources, "plot_script_sha256": sha(Path(__file__)),
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                "output_files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / "PLOT_MANIFEST.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for method in LABELS:
        values = np.asarray([[row["train_loss_mean"] for row in records[method, seed][0]] for seed in SEEDS])
        print(method, "loss_mean_epochs_1_2_50", values.mean(0)[[0, 1, 49]].tolist())
    print("PLOT_COMPLETE", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/training_checks/a0_b0_lp4_20261015")
    main(parser.parse_args().output)
