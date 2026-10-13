# 普通 CNN1D 对照：同深度无残差控制实验 V1

## 目标与学术边界

新增 **PlainCNN1D** 作为成熟 ResNet1D 的无 shortcut 对照；不是为了替换或隐藏已有强基线，也不是修改 NA-HSF/V5。A′ 已经是一个浅层 CNN1D，所以本次特意选择与既有 ResNet1D 尽可能接近的**深层普通 CNN**，避免重复跑一个功能重合的弱 CNN。

与已验证的 `FivePhaseResNet1D` 相同：原始五相 4096 点电周期、相同 `MDNormalization` (128 点)、5→32 kernel7 stem、每阶段2个双卷积块、32/64/128 通道、阶段过渡 stride2、GAP→128→6 分类器。**唯一主要结构调整**为移除 residual shortcut 与其跨通道1×1投影；因此参数量不会严格相同，需统计并如实声明。每块卷积和 BN 的其余顺序相似。不能据此强断言参数完全匹配或对残差作用进行严格的单因素因果辨识。

## 数据与冻结协议

- Source Train 34 recording / 738 cycles，Source Val 15 / 302；从已有 `v2_experiment` + `classical_baseline_experiment` 读取，绝不重新切分。
- Source 每轮 10/15/20/25/30/35/40 dB 整记录相电流 RMS 高斯增强，随机流由 seed 和 epoch 唯一确定；相同于已完成 ResNet 1D。
- Seeds 42/43/44；50 epochs、batch32、Adam 1e−3、weight_decay0；训练损失沿用原 `0.5 CE(clean)+0.5 CE(noisy)`。
- Source Val Clean 以及冻结的 0/5/10/20/30 dB。Clean Macro-F1 需距离该运行最佳 Clean 的 1pp 内，再按五档源验证平均 Macro-F1 最大、Clean 较好、epoch 较早选择。
- Target-Dev 88 recording / 每条件2474电周期，Clean、10/15/20/25/30/35/40 dB 原作者噪声，5/0 dB 按已冻结的 10 dB 残差放大生成（不是新采集的实测噪声）。任何目标数据不进入训练/选模。
- **禁止** Source Test 和 347 条 Final-Holdout 的训练、调参或中途评估。
- 输出单独保存在 `outputs/plain_cnn1d_control_v1`，绝不覆盖 `outputs/classical_baselines_v1`、V5、Xu-Full。

## 运行（服务器仓库根目录）

```bash
export SOURCE_STRONG_ROOT="/path/to/strong_data_root"  # 含generated/source
export TARGET_SNR_ROOT="/path/to/original_SNR_parent"
export TARGET_STRONG_ROOT="/path/to/strong_data_root"  # 含generated/target
bash scripts/run_plain_cnn1d_server.sh
```

独立步骤：`python scripts/plain_cnn1d_experiment.py smoke`、`train`、`evaluate`、`summarize`；三 seed 输出 `models/` 权重、`evaluations/` 逐样本预测、`reports/` Macro-F1/Accuracy/逐类F1-Recall与三 seed 均值±样本标准差。

## 结果解释与下一步

- 需要与**既有 ResNet1D** 和**现有 A′ / LP3 / LP4 / V5** 一起客观报告：如果无残差 CNN 与 ResNet 相当，说明强结果可能来自深度/通道/全局池化等因素，而非残差本身；若 ResNet 更高，才支持残差有潜力，但仍存在参数差异和训练动态混杂。
- 已有三 seed 实验显示 ResNet 在 Clean/10 dB 略高于 V5，但 V5 在 5/0 dB 较强；这是一个可研究的“信号物理约束 versus 大容量通用特征学习”的权衡，不应抹除 ResNet 已有结果。
- 是否将 residual adapter 加到 V5，**应基于新对照结果以及 0/5 dB 的真实收益**另行决策。先以固定深度、固定融合、固定 noise/seed/optimizer 的多臂消融区分容量和残差的作用。不能仅凭这项 CNN 对照保证 V5 加残差一定涨点。

本代码提交不等于已经得到服务器诊断成绩；正式运行和输出核验仍需在原服务器进行。
