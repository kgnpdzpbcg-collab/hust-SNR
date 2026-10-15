"""从已核验的完整六种子记录生成客观Markdown，不加入结果判断或调参建议。"""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(sys.argv[1]).resolve()
read = lambda path: json.loads(path.read_text(encoding="utf-8"))
deploy = read(ROOT / "DEPLOYMENT_MANIFEST.json")
config = deploy["config"]
status = read(ROOT / "RUN_STATUS.json")
verify = read(ROOT / "results/reports/VERIFICATION.json")
summary = read(ROOT / "results/reports/SUMMARY.json")["summary"]
assert status["status"] == "completed" and verify["status"] == "passed"
with (ROOT / "results/reports/PER_SEED_RESULTS.csv").open(encoding="utf-8") as stream:
    per_seed = list(csv.DictReader(stream))
with (ROOT / "results/reports/SOURCE_SELECTED_CHECKPOINTS.csv").open(encoding="utf-8") as stream:
    selected = list(csv.DictReader(stream))
names = {"resnet1d": "ResNet1D", "cnn1d_plain": "Plain CNN1D", "xu_full": "Xu-Full",
         "resnet_lp3": "ResNet＋LP3", "resnet_lp4": "ResNet＋LP4", "b0": "B0", "b1": "B1"}
conditions = ("clean", "snr10", "snr5", "snr0", "snr-5", "snr-10")


def score(method, condition):
    """只使用已通过独立复算的六种子均值及样本标准差。"""
    row = next(r for r in summary if r["method"] == method and r["condition"] == condition)
    return f"{row['macro_f1_mean']:.2f} ± {row['macro_f1_sd']:.2f}"


lines = ["# 新周期划分：七模型六种子服务器实验", "", "## Material Passport", "",
    "- Origin Skill: academic-research-suite / experiment-agent；Mode: run。",
    "- Date: 2026-10-15；Protocol: `hust-cycle-v2-seven-model-six-seed-v1`。",
    "- Verification: 保存权重的源验证和实测预测重放通过；未再次重训复现42组完整训练。", "",
    "## 代码、环境和执行", "",
    f"- 代码提交：`{deploy['code_commit']}`；分支`research/a1-b1-noise-structure-v1`。",
    "- 七个原模型类及各自损失直接复用，原算法文件没有修改；新增入口处理新数据、六种子和统一选模。",
    "- 服务器`wang@172.28.128.160`；RTX 5070 Ti；复用既有uv环境Python 3.12.3、PyTorch 2.14.1+cu130，没有安装依赖。",
    f"- 运行目录：`/home/wang/projects/会议论文/hust/experiments/source_cycle_benchmark/4728e9b`。",
    f"- 6进程并发，每进程2个CPU线程；42组训练、504组评价及汇总合计{status['elapsed_seconds']:.3f}秒，不含部署、预检和独立重放。",
    "- 各模型6个种子：42、43、44、2024、2025、2026。全部保留，没有按分数排除种子。", "",
    "## 数据与训练协议", "",
    "| 数据 | 原文件 | 周期数 | 用途 |", "|---|---:|---:|---|",
    "| 仿真Train | 63 | 962 | 更新参数 |", "| 仿真Val | 63 | 221 | 源域选模 |",
    "| 隔离周期 | 63 | 78 | 不参与训练/验证 |", "| 实测Dev | 88 | 每条件2474 | 源训练完成后评价 |",
    "| 实测Final-Holdout | 347 | 未制作 | 未读取波形、未评价 |", "",
    "- 数据协议`hust-source-cycle-split-v2`，SHA `4a0c76fd8a9c9ff6dbf40c7413a8ec50cbf71694b6dcf1f324e06a97337eed3d`。",
    "- 全63文件、13个工况均参与Train/Val；健康片段去重，各片段按时间分块，中间隔一个完整周期。相同记录的稳态周期仍有关联；该验证不代表独立采集的未见工况。",
    "- 59文件标签来自Excel，4个按用户授权从文件名推定并标明来源；400转/负载0仅有故障1、2、3。",
    "- 每样本为完整电周期，五相各4096点；Train/Val六类数量分别[146,166,166,166,157,161]和[38,37,37,37,36,36]。",
    "- 50轮、batch32、Adam lr=0.001、weight_decay=0，无scheduler。每轮31次更新，最后一批2个基础周期，配对后4个输入。",
    "- Train按整记录每相RMS加10/15/20/25/30/35/40 dB高斯噪声，每轮更新；同种子同轮次七模型使用相同噪声和采样顺序。",
    "- Val为Clean及0/5/10/20/30 dB固定噪声，七模型全部使用同一份数据。",
    "- 选模统一为：Clean F1在峰值1个百分点内，先选五噪声平均F1、再选Clean F1；完全并列时选五噪声平均CE较低者，最后选较早轮次。该规则相较部分旧实验增加CE平局判据。",
    "- ResNet、Plain CNN、LP3/LP4为原配对CE；Xu-Full为matched模式原CAE+DML，paper_sum重构及DML权重均1；B0为CE+0.2恢复项，B1为CE+1.0原复谱/相强度/噪声标定恢复项。",
    "- B0/B1各额外10轮恢复预训练，其余模型无额外预训练。",
    "- 实测条件为Clean、10/15/20/25/30/35/40/5/0/−5/−10 dB。全部采用同88个编号、同2474个周期，六类数量[1106,307,297,298,237,229]。",
    "- 5/0/−5/−10 dB复用作者10 dB残差缩放，不是新独立采集；88个Dev已用于此前方法开发，本轮不作为最终盲测。", "",
    "## 实测Dev Macro-F1", "", "单位为%，六种子均值±样本标准差（ddof=1）。标准差反映种子变化，不代表独立实测采集误差。", "",
    "| 模型 | Clean | 10 dB | 5 dB | 0 dB | −5 dB | −10 dB |", "|---|---:|---:|---:|---:|---:|---:|"]
for method in config["methods"]:
    lines.append("| " + names[method] + " | " + " | ".join(score(method, condition) for condition in conditions) + " |")
lines += ["", "其余15/20/25/30/35/40 dB及全部504行逐种子指标见CSV。", "", "## −10 dB逐种子Macro-F1", "",
    "| 模型 | 42 | 43 | 44 | 2024 | 2025 | 2026 |", "|---|---:|---:|---:|---:|---:|---:|"]
for method in config["methods"]:
    values = [float(next(r for r in per_seed if r["method"] == method and int(r["seed"]) == seed and r["condition"] == "snr-10")["macro_f1_percent"]) for seed in config["seeds"]]
    lines.append("| " + names[method] + " | " + " | ".join(f"{v:.2f}" for v in values) + " |")
lines += ["", "## 保存权重的选中轮次", "", "| 模型 | 42 | 43 | 44 | 2024 | 2025 | 2026 |", "|---|---:|---:|---:|---:|---:|---:|"]
for method in config["methods"]:
    values = [next(r for r in selected if r["method"] == method and int(r["seed"]) == seed)["selected_epoch"] for seed in config["seeds"]]
    lines.append("| " + names[method] + " | " + " | ".join(values) + " |")
lines += ["", "逐种子的首次六版本源验证满分轮次、满分轮数、选中CE和旧F1最早并列规则对应轮次均保存在SOURCE_SELECTED_CHECKPOINTS.csv。旧规则轮次只作记录，没有替换本轮best。", "",
    "## 核验和文件", "",
    f"- {verify['unit_tests_passed']}项服务器测试通过，42组六类真实源样本CUDA反传通过。",
    f"- 独立检查{verify['classification_epochs_checked']}轮分类历史、{verify['recovery_pretrain_epochs_checked']}轮恢复预训练，全部损失分项符合原公式。Xu的DML及总loss可以为负。",
    f"- 保存best重放{verify['source_validation_replays']}组源验证F1/CE；{verify['target_prediction_sets_reproduced_exactly']}组实测逐周期预测完全一致，{verify['summary_groups_independently_verified']}组六种子均值和标准差独立复算通过。",
    f"- {verify['protected_files_sha_unchanged']}份受保护文件SHA不变，{verify['new_weights_sha_unchanged']}份best/last权重SHA不变；推理前后参数及BN缓冲区逐张量一致。",
    "- [逐种子结果](results/reports/PER_SEED_RESULTS.csv) · [逐类F1/Recall](results/reports/PER_CLASS_PER_SEED.csv) · [12条件汇总](results/reports/SUMMARY.csv)",
    "- [选中轮次和验证满分记录](results/reports/SOURCE_SELECTED_CHECKPOINTS.csv) · [2100轮历史](results/reports/TRAINING_HISTORY.csv) · [恢复预训练](results/reports/PRETRAIN_HISTORY.csv)",
    "- [B1与六个基线逐种子配对](results/reports/PAIRED_B1_VS_BASELINES.csv) · [源验证重放](results/reports/SOURCE_VALIDATION_REPLAY.csv)",
    "- [部署清单](DEPLOYMENT_MANIFEST.json) · [核验结论](results/reports/VERIFICATION.json) · [导出文件SHA](RESULTS_EXPORT_MANIFEST.json)",
    "", "原始MAT、模型权重和环境不包含在Git发布记录中。历史结果未修改，本轮也未进行最终Holdout评价或显著性检验。"]
position = lines.index("## 核验和文件")
diagnostics = ["## 选中权重的源验证记录", "",
    "| 模型 | 六种子所选五噪声平均F1范围（%） | 出现过六版本全100%的种子数 |",
    "|---|---:|---:|"]
for method in config["methods"]:
    rows = [r for r in selected if r["method"] == method]
    values = [float(r["source_five_noise_mean_f1_percent"]) for r in rows]
    saturated = sum(int(r["six_conditions_100_epochs"]) > 0 for r in rows)
    diagnostics.append(f"| {names[method]} | {min(values):.2f}–{max(values):.2f} | {saturated}/6 |")
diagnostics.append("")
lines[position:position] = diagnostics
(ROOT / "SERVER_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("SERVER_RESULTS_WRITTEN", ROOT / "SERVER_RESULTS.md")
