# 七模型单种子t-SNE客观记录

特征/投影运行提交`d8194e85efc1ba3c7da384dce2ea336a7a0e7c4b`。日期2026-10-13。

## 展示种子选择

依据用户要求，仅选V5效果最好的共同种子；固定规则为V5在Clean、10、5、0 dB四条件平均Macro-F1最高，并列取较小种子。选择只用于展示，原源验证最佳权重和三种子正式成绩不改变。该选择使用Dev分数，是事后展示选择，不是独立测试或稳定性结论。

| 候选种子 | Clean | 10 dB | 5 dB | 0 dB | 四条件平均Macro-F1（%） |
|---:|---:|---:|---:|---:|---:|
| 42 | 96.58 | 96.57 | 95.89 | 92.47 | 95.38 |
| 43 | 98.55 | 96.92 | 95.44 | 89.96 | 95.22 |
| 44 | 99.54 | 99.03 | 96.92 | 90.46 | 96.49 |

**所选种子44用于全部七个模型、四个条件。** 不分别挑每个模型的好种子，也不声称44在每个单独SNR档位都是V5最高分。

## 数据、特征和投影

原88个Dev文件编号、每条件全部2474个周期，真实标签和顺序固定。前六模型使用352维编码特征；ResNet使用全局平均池化后的128维分类前向量，没有添加50维隐藏层。每张图显示全部周期，不删点或均衡采样；颜色/符号为真实类别。

全局RMS标量→PCA50→固定t-SNE；参数与上一轮完全一致：perplexity30、1500步、early_exaggeration12、learning_rate auto、Barnes-Hut angle0.5、随机状态20261013及相同二维高斯初始化。标签不参与投影拟合。

六模型Clean/10 dB共12份坐标原字节复用；新增六模型0/5 dB及ResNet四条件，共16份新投影；合计28份、四张七面板图。原空间指标在PCA前计算，5NN排除同一原始记录。

目标0/5 dB是既有作者10 dB噪声残差放大数据。四条件都是Dev记录的评估/分析，347个最终保留编号未读取。

## 各图与原分类器成绩（%，种子44单次结果）

| 模型 | Clean | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|
| A′无滤波 | 86.04 | 80.59 | 70.51 | 51.59 |
| 三阶低通 | 93.18 | 92.24 | 89.89 | 82.73 |
| 四阶低通 | 96.75 | 95.26 | 92.73 | 85.17 |
| 固定α=0.5 | 90.11 | 87.59 | 84.71 | 78.11 |
| V5 NA-HSF | 99.54 | 99.03 | 96.92 | 90.46 |
| Xu-Full | 70.05 | 64.95 | 53.73 | 32.17 |
| ResNet1D | 100.00 | 98.59 | 91.10 | 60.73 |

| 条件 | PNG（600 dpi） | PDF（嵌入字体、矢量散点） |
|---|---|---|
| Clean | [PNG](../../../figures/seven_model_tsne/server_d8194e8/encoder_clean_seed44.png) | [PDF](../../../figures/seven_model_tsne/server_d8194e8/encoder_clean_seed44.pdf) |
| 10 dB | [PNG](../../../figures/seven_model_tsne/server_d8194e8/encoder_snr10_seed44.png) | [PDF](../../../figures/seven_model_tsne/server_d8194e8/encoder_snr10_seed44.pdf) |
| 5 dB | [PNG](../../../figures/seven_model_tsne/server_d8194e8/encoder_snr5_seed44.png) | [PDF](../../../figures/seven_model_tsne/server_d8194e8/encoder_snr5_seed44.pdf) |
| 0 dB | [PNG](../../../figures/seven_model_tsne/server_d8194e8/encoder_snr0_seed44.png) | [PDF](../../../figures/seven_model_tsne/server_d8194e8/encoder_snr0_seed44.pdf) |

## 核验与解释边界

28组特征/坐标、69272行坐标标签和28组原空间几何指标独立复核通过；16组新增原预测与分类头重放完全一致，12组旧结果来自已核验可视化。重复ResNet的0 dB投影，最大坐标绝对差为0。模型提取前后状态一致，原结果/模型SHA不变。

四张图逐图查看并核查PNG尺寸、PDF文本边界、字体嵌入及28面板对齐；初次绘图检查发现图例与说明重叠，说明移到页脚后全部通过，不改变数据或坐标。

模型原生维度不同、各图独立拟合，不能比较跨图的坐标位置和绝对距离；特征轮廓系数与实际分类F1不等价。图是所选种子44的展示，不能替代三种子均值或以前六种子的稳定性记录。

[28组原空间指标](results/FEATURE_SEPARABILITY.csv) · [完整汇总](results/SUMMARY.json) · [展示选择原记录](results/DISPLAY_SEED_SELECTION.json) · [独立核验](results/VERIFICATION.json) · [图像核查](../../../figures/seven_model_tsne/server_d8194e8/VISUAL_QA.json) · [冻结协议](../../../docs/SEVEN_MODEL_TSNE_PROTOCOL.md)

发布二维坐标及指标，352/128维NPZ、模型权重和原始电流仅保存在服务器。
