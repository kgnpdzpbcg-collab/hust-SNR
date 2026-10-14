# LP4低阶谐波任务难度对照：服务器实验记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-14
- Verification Status: VERIFIED
- Version Label: lp4_task_difficulty_v1

## 执行配置

- 服务器`wang@172.28.128.160`；原uv/PyTorch环境复用，没有新增依赖下载。
- 新代码及执行前冻结协议提交`8be9c5a6c38ae316561dabbd96fbd9c5b3fdcd9e`。
- 新增LR、RBF-SVM、小型MLP，各42/43/44三个种子，共9份源域训练模型。
- 复用LP4＋A′ CNN（原`56513c7`）和LP4＋ResNet（原`5c75bea`）各三个原最佳权重，没有重训或重新选模。
- Source-Train/Val仍为34/15个仿真文件、738/302个周期；Target-Dev仍为88个实测文件、2474个周期。Source-Test、Final-Holdout和重复别名不用于训练、选模或推理。
- 新模型三个种子并发源训练，全体源选择与独立核验完成后才开始实测评估。训练噪声10/15/20/25/30/35/40 dB，50轮使用原噪声随机流；Val为Clean及0/5/10/20/30 dB。目标数据不参与缩放器拟合、候选选择或修改参数。

## 特征、模型与训练

五相电流按原Clean周期边界插值为4096点，经原LP4保留DC＋1–4阶，然后执行原MD中心化和低电流分支归一化。每相DC实数＋1–4阶实部/虚部，共45维，保留相位，不输入标签、文件编号、工况、高阶频谱或额外时域统计量。

全部源训练/验证及11种实测条件中，45维重建原LP4＋MD的5×128点网络输入最大绝对误差为`7.152557373046875e-7`，通过执行前冻结的2×10⁻⁶误差界限。

- LR/SVM每种子共享Clean＋50轮噪声的37638个训练视图。Clean每视图权重1，每个噪声视图权重1/50；两组总贡献比1:1。StandardScaler只在该源训练矩阵上按同权重拟合。
- LR：lbfgs、C∈{0.1,1,10}、tol=10⁻⁶、max_iter=2000，所有候选均无收敛警告。三个种子都选中C=10。
- SVM：RBF、C∈{0.1,1,10}、gamma=scale；三个种子都选中C=1，支持向量数分别12227/12255/12235。
- MLP：45→32→6、ReLU，1670参数，Adam lr=0.001、weight_decay=0、batch32，50轮1200次更新。损失0.5CE(Clean)+0.5CE(noisy)，沿用原样本排列随机流；选中轮次39/50/36。
- 所有模型仅在源Val选择：Clean距最大值不超过1个百分点，再取五档噪声Macro-F1均值最大，平分取Clean和较早候选/轮次。
- LR/SVM和神经网络的求解目标及优化次数不同；本实验对齐数据、噪声视图和源域选模规则，不记录为相同计算预算。

## 实测Dev Macro-F1

单位%，42/43/44均值±样本标准差（ddof=1）。完整11档逐种子指标另见CSV；5/0/−5 dB为作者10 dB残差缩放的合成压力条件。

| 方法 | Clean | 10 dB | 5 dB | 0 dB | −5 dB |
|---|---:|---:|---:|---:|---:|
| LP4＋LR | 53.50 ± 0.34 | 51.50 ± 0.07 | 48.63 ± 0.04 | 39.86 ± 0.03 | 30.46 ± 0.26 |
| LP4＋RBF-SVM | 64.60 ± 0.37 | 51.82 ± 0.83 | 20.15 ± 0.26 | 4.45 ± 0.13 | 2.91 ± 0.00 |
| LP4＋小MLP | 58.82 ± 1.69 | 54.46 ± 1.14 | 45.74 ± 1.55 | 34.47 ± 1.10 | 25.11 ± 1.24 |
| LP4＋A′ CNN | 96.28 ± 0.46 | 95.50 ± 0.47 | 93.69 ± 1.08 | 87.70 ± 2.38 | 73.23 ± 2.78 |
| LP4＋ResNet | 99.42 ± 0.50 | 99.00 ± 0.68 | 98.10 ± 0.85 | 94.44 ± 1.05 | 80.43 ± 2.92 |

## 源域已选模型成绩

Source-Train Clean为选中模型在738个源训练周期上的事后只读复算，未重新拟合。两个历史神经网络的Source-Train本轮未复算；其Source-Val来自原选中轮次，已有源验证重放记录。

| 方法 | Source-Train Clean Macro-F1 | Source-Val Clean Macro-F1 | Source-Val五档噪声平均Macro-F1 |
|---|---:|---:|---:|
| LP4＋LR | 94.73 ± 0.00 | 83.57 ± 0.00 | 70.11 ± 0.21 |
| LP4＋RBF-SVM | 100.00 ± 0.00 | 82.35 ± 0.18 | 53.68 ± 0.09 |
| LP4＋小MLP | 100.00 ± 0.00 | 83.05 ± 4.75 | 65.27 ± 2.65 |
| LP4＋A′ CNN | 本轮未复算 | 100.00 ± 0.00 | 98.43 ± 0.64 |
| LP4＋ResNet | 本轮未复算 | 100.00 ± 0.00 | 99.60 ± 0.14 |

小MLP第1/50轮训练损失分别为1.70042→0.02187、1.65154→0.01845、1.61783→0.01611。−5 dB时SVM三个种子的2474个周期均预测为标签4（OUS(A)&OUS(B)），Macro-F1为2.91%；真实六类数量及顺序未变。

源训练拟合及预测分布为事后描述，记录在[POSTHOC_SOURCE_FIT.json](results/reports/POSTHOC_SOURCE_FIT.json)。计算只读取源特征缓存与已保存模型/预测，没有改变特征、参数、已选权重或实测评估结果。

## 核验、时间与保存位置

- [服务器四项数值测试](unit_tests.log)通过：信息重建、先滤波后归一化、高阶去除、相位保留及小MLP预算。
- [实测访问前独立核验](PRE_TARGET_SOURCE_VERIFICATION.json)：9份源选择独立复算、9份加权Scaler均值/方差独立复算、54组已选源预测复现。
- [评估核验](results/reports/VERIFICATION.json)：165组指标独立复算；99组新模型预测重复一致；63组历史CNN/ResNet预测逐周期完全复现；数据编号、顺序、标签不变，参考模型参数及BN缓冲值不变。
- 30个新模型/Scaler/运行记录及参考权重文件，在实测评估和事后复算前后SHA不变；4329个旧文件指纹未变。
- 三个种子的训练用时分别54.899/55.297/54.536秒，实测评估及核验71.840秒；[整个调度用时137.168秒](RUN_STATUS.json)，另附事后源拟合复算。
- 服务器独立目录：`/home/wang/projects/会议论文/hust/experiments/lp4_task_difficulty/20261014/`；权重位于`results/models/`，共享特征缓存位于`results/cache/`。

## 文件索引

- [执行前协议快照](code/docs/LP4_TASK_DIFFICULTY_V1.md)、[配置](code/configs/lp4_task_difficulty_v1.json)、[来源](DEPLOYMENT_MANIFEST.json)、[训练命令与PID](TRAINING_COMMANDS.json)、[评估命令](EVALUATION_COMMAND.json)。
- [165行逐种子结果](results/reports/PER_SEED_RESULTS.csv)、[990行六类F1/Recall](results/reports/PER_CLASS_PER_SEED.csv)、[55组未舍入汇总](results/reports/SUMMARY.json)。
- [逐模型预测与混淆矩阵](results/evaluations/)、[候选参数/MLP轨迹](results/models/)、[各种子50轮源噪声计划与源指纹](results/reports/)。
- [原发布字节指纹](RESULT_EXPORT_MANIFEST.json)、[事后描述发布指纹](POSTHOC_EXPORT_MANIFEST.json)。GitHub没有上传波形MAT、分类器权重、特征缓存或环境。

本记录只覆盖该45维表示、固定参数范围、三个种子及仿真→实测Dev协议；没有评估其他特征定义、额外调参预算或最终保留集。
