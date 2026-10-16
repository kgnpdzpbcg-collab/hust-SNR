# 新增4/−4/−8 dB：42/44固定权重服务器实验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；scientific-figure-making。
- Date: 2026-10-15；Mode: 生成、固定权重推理和表示可视化。
- Verification: 数据独立核查、24组预测重放、24组特征/坐标核查通过；没有重新训练。

## 原代码、权重及目录

- 原训练提交4728e9b，新划分Train962/Val221；四模型42/44共8份best，原epoch未重新选择。
- 服务器目录：/home/wang/projects/会议论文/hust/experiments/additional_snr_seed42_44/20261015_snr4_m4_m8。
- 数据位于data/SNR_4、data/SNR_-4、data/SNR_-8；新评价在evaluation，特征/坐标在tsne，与旧五档及六种子目录分开。
- 复用既有uv/PyTorch环境，不安装依赖。原模型及损失不改；本次没有训练或源验证选模。

## 第一步：数据生成与独立核验

- 固定88个实测Dev编号，每档88个文件，共264个MAT。没有读取最终Holdout。
- 每份MAT为I_noisy字段、10031×7的float64；时间和角速度列与Clean逐值相同。
- I_new = I_clean + (I_author10 − I_clean) × 10^((10−目标dB)/20)。沿用原缩放函数，不添加另一份高斯噪声。
- 4/−4/−8 dB倍数为1.9952623149688795、5.011872336272722、7.943282347242816。
- SNR按整条原记录逐相计算，不表示每个截取周期均精确达到指定SNR；三档共享同一作者噪声残差，不是独立采集。
- 保存MAT后独立重算能量比及残差比例；整记录逐相SNR最大误差3.863576125695545e-14 dB。176份原输入SHA未变，264份新MAT经评价后SHA未变。

## 第二步：固定权重评价

每档使用同2474个周期、五相各4096点，沿用原时间合并/周期边界/标签和各模型自己的预处理。六类数量[1106,307,297,298,237,229]。以下Macro-F1单位%，每行保留一个实际种子。

| 模型 | 种子 | 4 dB | −4 dB | −8 dB |
|---|---:|---:|---:|---:|
| ResNet＋LP4 | 42 | 97.99 | 84.47 | 67.83 |
| ResNet＋LP4 | 44 | 94.78 | 79.20 | 65.20 |
| Plain CNN1D | 42 | 90.44 | 38.84 | 20.48 |
| Plain CNN1D | 44 | 84.94 | 39.28 | 16.88 |
| ResNet1D | 42 | 83.42 | 29.94 | 13.85 |
| ResNet1D | 44 | 83.34 | 26.38 | 13.80 |
| B1 | 42 | 97.72 | 94.34 | 81.39 |
| B1 | 44 | 99.45 | 93.18 | 82.00 |

24组预测均在重新加载固定best后重放一致。Accuracy、各类F1/Recall及故障漏报/健康误报见CSV，混淆矩阵及逐周期预测见evaluation/evaluations。

## 第三步：分类前128维t-SNE

- 四模型统一钩取六分类Linear输入，即GAP后的128维向量；B1和LP4完整执行各自原forward，不跳过前端。
- 24组分类头logits逐值重放一致，分类预测与第二步正式预测逐项一致。所有原84份best/last权重SHA保持不变。
- 全局RMS单标量→PCA50不白化→固定t-SNE；perplexity30、1500轮、random_state20261013、同一高斯二维初值，与前五档参数完全相同。PCA50保留方差99.6735%–99.9981%。
- 每档独立两行四列：上42、下44；列LP4、CNN1D、ResNet1D、B1。每面板全部2474点，颜色为真实标签，降维拟合不使用标签。
- 各面板独立拟合；绝对坐标和簇间距离不能跨图作数值比较。42/44是此前看过六种子结果后由用户指定的展示子集，不替代六种子稳定性记录。

| 条件 | 600dpi PNG | 字体嵌入PDF |
|---|---|---|
| 4 dB | [PNG](tsne/figures/tsne_snr4_seeds42_44.png) | [PDF](tsne/figures/tsne_snr4_seeds42_44.pdf) |
| -4 dB | [PNG](tsne/figures/tsne_snr-4_seeds42_44.png) | [PDF](tsne/figures/tsne_snr-4_seeds42_44.pdf) |
| -8 dB | [PNG](tsne/figures/tsne_snr-8_seeds42_44.png) | [PDF](tsne/figures/tsne_snr-8_seeds42_44.pdf) |

图片使用本地已有.cache/figure_runtime导出；服务器不安装matplotlib。180×112mm，三张PNG/PDF均人工查看并检查24面板、图例、页脚无重叠，PDF字体均嵌入。

## 阶段、环境及文件

六阶段按generate→verify-data→evaluate→verify-evaluation→extract→project依次执行，每阶段通过后才启动下一阶段，合计48.598秒，不含部署、本地绘图与最终整理。原uv环境与前轮相同，没有重新下载PyTorch。

[24行逐种子指标](evaluation/PER_SEED_RESULTS.csv) · [144行各类F1/Recall](evaluation/PER_CLASS_PER_SEED.csv) · [24组特征几何指标](tsne/FEATURE_SEPARABILITY.csv)

[数据生成清单](data/GENERATION_MANIFEST.json) · [数据独立核验](data/VERIFICATION.json) · [预测重放核验](evaluation/VERIFICATION.json) · [最终核验](VERIFICATION.json) · [图像核验](FIGURE_QA.json) · [阶段命令与用时](STAGE_STATUS.json)

本地记录不含原始/生成MAT、模型权重或高维NPZ；它们留在服务器。此前所有结果、五档图片及六种子记录保持原样。
