# LP4低阶谐波任务难度对照协议

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-14
- Verification Status: UNVERIFIED（执行前协议；实际核验状态写入服务器记录）
- Version Label: lp4_task_difficulty_v1

## 冻结输入与对照

新增Logistic Regression、RBF-SVM、小型MLP；复用LP4＋A′ CNN与LP4＋ResNet的原42/43/44最佳权重。原34/15仿真Train/Val、738/302个周期以及88条实测Dev、2474个周期均保持不变。不使用Source-Test、实测Final-Holdout或重复别名训练/评估。

每个完整周期先按原规则插值为5×4096点，应用原LP4（保留DC及1–4阶），再执行原MD的极值中心化、低电流相除100/其他相除峰值。提取每相DC实数及1–4阶复系数实部/虚部，共45维。不添加工况、编号、标签、高阶频谱或时域统计量。核验45维表示可在浮点误差2×10⁻⁶内重建原LP4＋MD的5×128点网络输入。

## 训练与参数选择

- 三个种子42/43/44；所有新分类器从头拟合。
- 训练噪声仅10/15/20/25/30/35/40 dB，使用原`fresh_noisy_source(seed,epoch=1..50)`的同一50轮随机流。固定Val为Clean以及0/5/10/20/30 dB；0/5直接使用此前已核验的源域MAT。
- 每个种子共享Clean及50轮噪声特征。LR/SVM合计37638个视图，Clean权重1，每个噪声视图1/50，使两部分总权重1:1；StandardScaler同样只在此源训练集上加权拟合，不使用Val或Target统计量。
- LR：multinomial默认六类softmax，lbfgs，C∈{0.1,1,10}，最多2000次迭代，tol=10⁻⁶；出现收敛警告立即记录并停止。
- RBF-SVM：C∈{0.1,1,10}，gamma=scale，缓存512 MB；同种子各候选接收同一矩阵和权重。
- MLP：45→32→6，ReLU，1670参数，50轮、batch32、Adam lr=0.001、weight_decay=0；原样本排列随机流，损失0.5CE(Clean)+0.5CE(noisy)，共1200次更新。
- 候选/轮次先满足Clean距最大源验证Macro-F1不超过1个百分点，再取五档源噪声验证Macro-F1均值最大，平分时按Clean及较早候选/轮次排序。
- LR/SVM的求解目标、优化次数与神经网络不同；本协议对齐标签、数据视图、噪声贡献及选模规则，不声称计算预算完全相同。
- 三个种子的全部源域选择完成后，才启动实测Dev评估；不根据实测结果改特征、参数或种子。

## 固定实测评估

统一评估Clean、10/15/20/25/30/35/40/5/0/−5 dB共11条件。0/5/−5属于作者10 dB残差缩放得到的合成压力条件，噪声形态相关，不视为独立采集。

全部5种方法×3种子×11条件，共165组结果；保留Macro-F1、Accuracy、六类F1/Recall及混淆矩阵。报告均值和样本标准差，不将同文件周期或3种子当成大量独立受试样本进行显著性检验。已有CNN/ResNet十条件预测及ResNet−5 dB预测须逐周期复现。新模型重复推理须一致，标准化器和权重在评估前后SHA不得变化。

## 文件与执行边界

[配置](../configs/lp4_task_difficulty_v1.json)、[特征/小MLP](../scripts/lp4_difficulty_models.py)、[独立实验入口](../scripts/lp4_difficulty_experiment.py)、[数值测试](../tests/test_lp4_difficulty.py)。服务器复用原uv/PyTorch环境，代码和输出写入独立实验目录；原模型和冻结实验配置保持不变。

结果只描述当前LP4/MD表示、固定数据划分及仿真→实测Dev条件下的分类表现。不预先把简单模型的高分解释为所有诊断任务容易，也不把低分直接解释为深度ResNet必不可少。
