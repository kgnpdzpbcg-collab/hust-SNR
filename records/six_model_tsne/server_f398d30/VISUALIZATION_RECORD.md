# 六模型固定权重特征可视化记录

运行代码提交：`f398d308f046f4b5bd7e001888007fbf85669556`。日期：2026-10-13。

完成六模型、共同种子42/43/44、Clean/10 dB、编码器352维/分类隐藏层50维的72组特征投影，导出12张六面板图。使用已有源验证最佳权重，未训练、更新参数或重新选模。

## 数据、权重与特征层

- 模型：A′无滤波、三阶低通、四阶低通、固定α=0.5、V5 NA-HSF、Xu-Full。
- Xu-Full当前仅有42/43/44三个种子，因此本次六模型采用共同的42/43/44；其他五模型的2024/2025/2026结果仍在原记录中，不属于本次共同种子汇总。
- 数据：冻结Target-Dev的88个实测文件编号，每个条件全部2474个完整电周期。六模型的样本、真实标签和顺序一致，未按类别均衡删点。
- 六类周期数依次为1106/307/297/298/237/229；类别为Healthy、OUS(A)、OLS(A)、OP(A)、OUS(A)&OUS(B)、OLS(A)&OP(C)。
- Clean读取原干净实测电流；10 dB读取原作者SNR目录下相同88个编号。沿用各模型正式评估的滤波、MD归一化及128点重采样，没有新增加噪声或修改输入处理。
- 主图为编码器卷积输出展开后的352维特征；补充图为最终六类Linear层前的50维隐藏特征。Xu-Full的DML作用于352维编码特征。六类logits不作为本次可视化特征。
- 权重、特征、二维文件SHA及原权重路径见[特征清单](results/FEATURE_MANIFEST.json)。347个Final-Holdout编号未读取或评估。

## 投影及指标定义

每组先除以一个由全部特征元素计算的全局RMS标量，再PCA降至50维，最后拟合二维t-SNE。全局标量不改变欧氏几何；不做逐维或逐样本标准化。标签不参与尺度处理、PCA或t-SNE拟合。

固定参数：perplexity=30，early_exaggeration=12，learning_rate=auto，1500步预算，Barnes-Hut angle=0.5，random_state=20261013。所有组使用同一2474×2高斯初始坐标，标准差0.0001。各模型、条件和层独立拟合，不将不同网络的未对齐神经元联合降维。

原空间指标均在PCA之前计算：

- **宏平均轮廓系数S**：先计算每个周期的欧氏轮廓系数，再分别按六类求均值，最后六类等权平均；范围−1至1。图标题显示此S。
- **跨记录5NN同类率**：对每个周期排除来自同一原始MAT文件的全部周期，取其他记录中最近5个周期，计算真实标签相同的比例，最后按六类等权平均。该指标使用Dev真实标签做事后几何分析，不是原分类器的Accuracy或新的测试成绩。
- **Macro-F1**：原模型分类头实际预测的六类F1等权平均，没有用t-SNE坐标建立新分类器。
- 另外保存各类轮廓系数、各类5NN同类率、PCA保留方差、t-SNE可信度k=15、KL和迭代数。

以下统计均为共同三个训练种子的均值±样本标准差（ddof=1）；F1和同类率以百分数表示，S保持原数值。

## 编码器352维：主图对应原空间指标

| 模型 | Clean S | 10 dB S | Clean跨记录5NN同类率（%） | 10 dB跨记录5NN同类率（%） |
|---|---:|---:|---:|---:|
| A′无滤波 | 0.151 ± 0.042 | 0.104 ± 0.028 | 96.09 ± 1.01 | 94.93 ± 1.18 |
| 三阶低通 | 0.183 ± 0.028 | 0.170 ± 0.028 | 96.91 ± 0.14 | 96.87 ± 0.20 |
| 四阶低通 | 0.163 ± 0.019 | 0.147 ± 0.019 | 96.32 ± 0.69 | 96.30 ± 0.38 |
| 固定α=0.5 | 0.175 ± 0.042 | 0.162 ± 0.041 | 96.48 ± 0.17 | 96.47 ± 0.31 |
| V5 NA-HSF | 0.199 ± 0.030 | 0.183 ± 0.032 | 97.12 ± 0.26 | 96.99 ± 0.10 |
| Xu-Full | 0.055 ± 0.005 | 0.032 ± 0.006 | 95.10 ± 0.07 | 93.28 ± 0.27 |

## 分类隐藏层50维：补充图对应原空间指标

| 模型 | Clean S | 10 dB S | Clean跨记录5NN同类率（%） | 10 dB跨记录5NN同类率（%） |
|---|---:|---:|---:|---:|
| A′无滤波 | 0.429 ± 0.089 | 0.322 ± 0.087 | 99.53 ± 0.75 | 97.18 ± 1.67 |
| 三阶低通 | 0.470 ± 0.061 | 0.449 ± 0.065 | 99.36 ± 1.09 | 99.14 ± 0.76 |
| 四阶低通 | 0.453 ± 0.048 | 0.428 ± 0.051 | 99.84 ± 0.24 | 99.43 ± 0.49 |
| 固定α=0.5 | 0.485 ± 0.070 | 0.464 ± 0.072 | 99.38 ± 0.54 | 99.08 ± 0.74 |
| V5 NA-HSF | 0.513 ± 0.060 | 0.491 ± 0.064 | 100.00 ± 0.00 | 99.77 ± 0.12 |
| Xu-Full | 0.187 ± 0.020 | 0.132 ± 0.018 | 96.65 ± 0.75 | 93.79 ± 1.21 |

## 原分类器Macro-F1：同三种子预测重放

| 模型 | Clean Macro-F1（%） | 10 dB Macro-F1（%） |
|---|---:|---:|
| A′无滤波 | 89.10 ± 2.68 | 85.22 ± 4.87 |
| 三阶低通 | 96.64 ± 3.05 | 95.75 ± 3.06 |
| 四阶低通 | 96.28 ± 0.46 | 95.50 ± 0.47 |
| 固定α=0.5 | 95.62 ± 4.94 | 93.85 ± 5.55 |
| V5 NA-HSF | 98.22 ± 1.51 | 97.51 ± 1.33 |
| Xu-Full | 71.56 ± 2.10 | 64.64 ± 1.02 |

## 全部图像

每张图包括六个模型、全部2474个周期，颜色和符号表示真实类别；F1与S分别来自原分类器和原特征空间。种子42只作为固定入口图，全部42/43/44均发布，不按照得分选择入口。PNG为600 dpi，PDF嵌入字体、保留矢量散点。

| 层/条件 | 种子42 | 种子43 | 种子44 |
|---|---|---|---|
| 编码器352维 / Clean | [PNG](../../../figures/six_model_tsne/server_f398d30/encoder352_clean_seed42.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/encoder352_clean_seed42.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/encoder352_clean_seed43.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/encoder352_clean_seed43.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/encoder352_clean_seed44.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/encoder352_clean_seed44.pdf) |
| 编码器352维 / 10 dB | [PNG](../../../figures/six_model_tsne/server_f398d30/encoder352_snr10_seed42.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/encoder352_snr10_seed42.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/encoder352_snr10_seed43.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/encoder352_snr10_seed43.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/encoder352_snr10_seed44.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/encoder352_snr10_seed44.pdf) |
| 分类隐藏层50维 / Clean | [PNG](../../../figures/six_model_tsne/server_f398d30/classifier50_clean_seed42.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/classifier50_clean_seed42.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/classifier50_clean_seed43.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/classifier50_clean_seed43.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/classifier50_clean_seed44.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/classifier50_clean_seed44.pdf) |
| 分类隐藏层50维 / 10 dB | [PNG](../../../figures/six_model_tsne/server_f398d30/classifier50_snr10_seed42.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/classifier50_snr10_seed42.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/classifier50_snr10_seed43.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/classifier50_snr10_seed43.pdf) | [PNG](../../../figures/six_model_tsne/server_f398d30/classifier50_snr10_seed44.png) · [PDF](../../../figures/six_model_tsne/server_f398d30/classifier50_snr10_seed44.pdf) |

## 执行与验证

- 特征提取和降维在`wang@172.28.128.160`既有uv/PyTorch环境执行；四个CPU降维进程、每进程四个线程。
- 特征提取6.991秒；降维65.534秒；两阶段合计墙钟72.525秒。未重新训练。
- 36组原分类预测与历史保存预测完全一致；两层特征分别通过原分类头重放，共72组完全一致。提取前后所有模型状态张量相同。
- 核查72份特征、72份二维坐标文件、178128行坐标及标签，独立重算F1和24组汇总。
- 对A′种子42/Clean/352维的投影额外重复计算，二维坐标最大绝对差为0；该重复不改变正式汇总。
- 本地绘图复用既有`.cache/figure_runtime` Matplotlib 3.11.2；核查24个PNG/PDF的SHA、12份PDF页面/嵌入字体/文本边界，以及72个面板尺寸和对齐。全部12张图已人工查看，标题、图例和页脚未发现裁切或遮挡。
- 服务器导出文件逐个SHA核对，历史权重和原结果保持不变。原始电流、MAT、PT权重、352/50维NPZ不上传；发布的CSV仅包含二维坐标、记录编号、真实标签和原分类预测。
- 初次目录`fc4c487`在CPU/CUDA状态相等性检查处失败，保留原日志；修正比较设备后，正式运行使用独立新目录`f398d30`。原模型及输入代码不变。

[执行状态](RUN_STATUS.json) · [独立复核](results/VERIFICATION.json) · [72组逐种子指标](results/FEATURE_SEPARABILITY_PER_SEED.csv) · [完整汇总](results/SUMMARY.json) · [图像清单](../../../figures/six_model_tsne/server_f398d30/FIGURE_MANIFEST.json) · [图像检查](../../../figures/six_model_tsne/server_f398d30/VISUAL_QA.json)

## 解释边界

二维t-SNE可能改变距离和簇形状；不同面板的坐标系、整体位置及绝对距离不可直接比较。相同颜色的小簇不必对应一个整体簇，工况和记录也会形成局部结构。轮廓系数和跨记录5NN同类率描述特征几何，与原分类器决策边界并不等价。本记录仅使用共同三个种子的冻结Dev集，不是Final-Holdout评估，也不覆盖另外三个种子的稳定性。

复现入口、参数、限制见[冻结协议](../../../docs/SIX_MODEL_TSNE_PROTOCOL.md)。
