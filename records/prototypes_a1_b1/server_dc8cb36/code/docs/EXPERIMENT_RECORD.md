# HUST SNR实验记录：gate0_v1

实验日期：2026-10-12。训练条件为`baseline`和`augmentation`，种子42/43/44。共6个模型、18次源测试/目标开发集评估。

## 数据划分

| 域/用途 | 原始记录 | 周期样本 | 类别0–5的周期数量 |
|---|---:|---:|---|
| 仿真Train | 34 | 738 | 126 / 127 / 127 / 127 / 104 / 127 |
| 仿真Val | 15 | 302 | 42 / 52 / 52 / 52 / 52 / 52 |
| 仿真Source-Test | 10 | 174 | 24 / 30 / 30 / 30 / 30 / 30 |
| 实测Dev：Clean | 88 | 2474 | 1106 / 307 / 297 / 298 / 237 / 229 |
| 实测Dev：10 dB | 88，同一批记录 | 2474，同一批窗口 | 1106 / 307 / 297 / 298 / 237 / 229 |
| 实测Final-Holdout | 347 | 本轮未生成推理结果 | 本轮未评估 |

仿真59条标注记录按12个转速/负载组分成7/3/2组。实测440条标注记录按完整五相电流相同的记录选择字典序首条，得到435条代表记录；其中88条为Dev、347条为Holdout。分组在模型训练之前固定。

## 训练配置

- 模型：五通道1D-CNN，Conv通道5→16→32→32，kernel16/4/4，BN/LeakyReLU/MaxPool；FC352→50→6；25,620个参数。
- 输入：五相电流，完整电气周期，128点。健康窗口0.1–0.45s，故障窗口从0.5s后两个电气周期开始，结束边界0.98s。
- 归一化：各相减去极值中心；平均绝对值≥0.2时除以最大绝对值，否则除以100；随后线性重采样。
- Adam，lr=0.001，weight_decay=0，batch=32，50个完整epoch，每模型1200次更新，无scheduler或early stopping。
- 最佳权重依据仿真Clean-Val Macro-F1最大值选择，相同值取最早epoch。
- baseline：仿真干净视图CE。augmentation：0.5 CE(干净视图)+0.5 CE(含噪视图)。
- 增强对仿真Train整记录逐相添加Gaussian噪声，按整记录RMS校准；每轮从10/15/20/25/30/35/40 dB中抽取。目标域数据不参与参数或BN训练。
- 三进程并发，每进程4CPU线程；确定性FP32，TF32关闭。训练环境复用，未创建新训练环境。

## 环境与执行记录

- NVIDIA GeForce RTX 5070 Ti，16 GB。Python3.12.3、Torch2.14.1+cu130、NumPy2.5.3、SciPy1.18.1。
- baseline训练批次窗口：5.295928秒；augmentation：8.675042秒。窗口含进程启动及模型读写，不含数据整理、传输、独立核验、评估或文档生成。

## 汇总指标

单位：%，3个种子的均值 ± 样本标准差（ddof=1）。

| 条件 | 数据集 | Macro-F1 | Accuracy | Balanced Accuracy |
|---|---|---:|---:|---:|
| baseline | source_source_test | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| baseline | target_dev_clean | 96.77 ± 2.55 | 97.43 ± 2.42 | 96.89 ± 2.31 |
| baseline | target_dev_snr10 | 78.50 ± 7.57 | 81.26 ± 9.62 | 80.45 ± 6.45 |
| augmentation | source_source_test | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| augmentation | target_dev_clean | 95.08 ± 1.71 | 95.07 ± 3.24 | 96.46 ± 0.33 |
| augmentation | target_dev_snr10 | 83.73 ± 7.25 | 83.31 ± 9.34 | 86.48 ± 4.96 |

## 配对数值差

单位：Macro-F1百分点。按相同种子作差。

| 差值定义 | seed42 | seed43 | seed44 | 均值 | 标准差 | 记录组bootstrap 95%区间 |
|---|---:|---:|---:|---:|---:|---|
| Baseline Clean − Baseline 10 dB | +11.91 | +22.63 | +20.28 | +18.27 | 5.64 | [+15.33, +22.03] |
| Augmentation 10 dB − Baseline 10 dB | -0.21 | +14.30 | +1.63 | +5.24 | 7.90 | [+2.68, +8.02] |
| Augmentation Clean − Baseline Clean | -3.95 | -0.37 | -0.76 | -1.69 | 1.97 | [-3.71, +0.29] |

Bootstrap为5000次，随机种子20261012；重抽单位为88条基础记录，同记录的所有周期、两个目标条件、三个模型种子一起重抽。该区间的计算固定本轮模型与Dev记录。

## 权重选择与核验

| 条件 | seed | 最佳epoch | Val Macro-F1 | 更新次数 |
|---|---:|---:|---:|---:|
| baseline | 42 | 3 | 100.00% | 1200 |
| baseline | 43 | 3 | 100.00% | 1200 |
| baseline | 44 | 5 | 100.00% | 1200 |
| augmentation | 42 | 2 | 100.00% | 1200 |
| augmentation | 43 | 3 | 100.00% | 1200 |
| augmentation | 44 | 3 | 100.00% | 1200 |

- 六个模型的初值、源Val最佳选择、BN计数已核对；两条件同种子的初值及样本顺序一致。
- 18次评估重新创建模型、重载最佳权重，分数与预测逐位一致；推理未改变模型参数或BN。
- 首次环境指纹比较涉及Python tuple与JSON list格式差异；格式规范化后字段一致。修正前后SHA在VERIFIER_REPAIR.json中，模型未重训，预测未修改。
- Final-Holdout未评估。原始MAT、压缩包、模型权重、训练环境和中间特征NPZ不在此仓库。
- [逐种子CSV](../records/gate0_v1/PER_SEED_RESULTS.csv)、[汇总JSON](../records/gate0_v1/SUMMARY.json)、[冻结协议](../configs/protocol_gate0_v1.json)、[划分清单](../configs/splits_gate0_v1.json)。

## 文件导出

发布记录中的SUMMARY删除原文件的conclusion字段，BASELINE_GATE删除interpretation字段；数值、预测、历史与冻结配置保持原值。原服务器报告生成器及统计解释文档未发布。原字节映射及SHA256记录在[PUBLICATION_MANIFEST.json](../records/PUBLICATION_MANIFEST.json)。

发布前用逐条真值和预测重算18次评估的三个指标，并核对6组汇总、3组配对差值及36个图文件哈希；核对记录见[PUBLICATION_VERIFICATION.json](../records/PUBLICATION_VERIFICATION.json)。
