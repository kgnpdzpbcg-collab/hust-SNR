# HUST-SNR V5：A′ + Noise-Aware Harmonic Selective Fusion（NA-HSF）

**状态：运行提交ae2716b已完成GPU三种子50轮训练及30组Target-Dev评估，36项测试、12组源验证指标、30组预测/门控统计和60行历史配对独立复核通过。** 独立研究分支：research/na-hsf-v5。只读取冻结 Source Train/Val 和 88 条 Target-Dev；347 条 Final-Holdout 继续隔离。完整记录见[服务器结果](../records/v5_nahsf/server_ae2716b/SERVER_RESULTS.md)。

## 1. 研究假设与数据证据

- V4 原始训练协议：A′+LP3 的 10 dB 目标 Dev Macro-F1 = 95.79%；A′+LP4 = 96.63%（三种子均值，记录：records/v4_hpasf/server_b4801df）。
- 旧权重合成 0 dB 压力测试：LP3 = 88.51%，LP4 = 87.91%（记录：records/v4_strong_snr/server_84f6757）。这表明更强去噪和更多谐波信息的取舍随噪声条件变化，但不能据此保证学习式门控有用。
- HPASF V0 将第4阶基础增益限制在约 0.05，训练后基本不变，并出现随机种子性能崩溃。NA-HSF 改为从0.5初始化的三描述符输入动态门控，不再设置0.05的乘性上界。
- 方法构思阶段已分析 88 条 Dev 波形。**仅训练权重和选 checkpoint 完全基于 Source；不能声称整个方法研发对目标域完全盲检。**

## 2. 严格的算法定义

输入为五相原始电流、完整电气周期，均匀插值到 4096 个点，张量形状 [B,5,4096]。在 MD 归一化之前计算 rFFT，使用 norm="ortho"。第 k 阶的五相平均功率定义为：

\[
P_k=\frac{1}{5}\sum_{p=1}^{5}|X_{p,k}|^2.
\]

统计前1–3阶、第1–30阶和第12–30阶的功率：

\[
P_{low}=\sum_{k=1}^3 P_k,\qquad P_{total}=\sum_{k=1}^{30}P_k,\qquad
P_{high}=\sum_{k=12}^{30}P_k.
\]

频谱噪声底代理：第12–30阶的19个功率值的中位数。在 CUDA 上以 stable sort 后第十个元素实现，避免带索引的 median 在严格确定性模式下异常。

\[
\widehat N=\mathrm{median}_{k=12,\ldots,30}P_k.
\]

门控输入固定为三个**来自当前含噪输入、完全无标签**的描述符：

\[
z_1=\frac{P_{low}}{\max(P_{total},\epsilon)}, \quad
z_2=\frac{\mathrm{clip}(\log_{10}\frac{\max(P_4,\epsilon)}{\max(\widehat N,\epsilon)},-6,6)}{6},
\quad
z_3=\frac{P_{high}}{\max(P_{total},\epsilon)}.
\]

其中 eps 取对应浮点 dtype 的 torch.finfo(...).eps。z2 做了 log10 压缩与截断，避免原始功率比在不同噪声强度下范围过大；它是**可靠性代理，不是精确的目标 SNR**。描述符本身 no_grad，门控输出对下游 CE 保留梯度。

门控神经网络：

    Linear(3,8) → ReLU → Linear(8,1) → Sigmoid

最后的 Linear 的 weight 和 bias 全零初始化，因此最初每个样本 α=0.5，整个频域模块从 LP3 与 LP4 等权开始。第一个优化步只有门控末层接收梯度，随后首层能通过学习过的末层接收梯度。**A′ CNN 在 Gate MLP 之前实例化**，同种子时保持旧 V4 的 CNN 初始化字节不变。

频域整形精确定义：

\[
\widetilde X_k=\begin{cases}
X_k & k=0,1,2,3\\
\alpha(x)X_4 & k=4\\
0 & k\ge5.
\end{cases}
\]

然后 iRFFT → MD 极值居中和弱电流相缩放 → 重采样至五相×128 → A′ 原 25,620 参数 CNN → 六类输出。各相共享同一个实数 α，只改变第4阶幅值，不主动旋转谐波相位，不进行五相混合。

**重要的机制边界**：上述整形在数学上等价于 (1−α) LP3(x)+α LP4(x)。这不是两个独立的 CNN，也不是原始 Selective Kernel Networks 的逐层复现。借鉴的是输入自适应选择思想；具体频谱质量描述和一维五相频率整形是任务适配。

## 3. 固定实验协议（与 V4 原始协议严格对照）

| 项目 | V5 设置 |
|---|---|
| 源域 Train / Val | Gate0 原冻结 34 / 15 条记录，738 / 302 周期 |
| 源域 Test | 原冻结10条，本轮未评估、不选模 |
| 训练噪声 | 10、15、20、25、30、35、40 dB；整记录逐相 RMS 校准高斯，每轮重新生成 |
| 源域固定验证 | Clean 和 10、20、30 dB |
| checkpoint 选择 | 所有 epoch 的最高 Clean F1 容差1个百分点，选固定三档带噪 Macro-F1 均值最大；再优先 Clean、较早 epoch |
| 优化器 | Adam，lr=0.001，weight_decay=0 |
| 训练预算 | 50 epochs，batch32，seed 42/43/44 |
| 分类损失 | 0.5 CE(Clean)+0.5 CE(noisy)，没有额外频谱正则/重构/一致性损失 |
| 模型 | 固定 A′ CNN 25,620 参数，新增 Gate MLP 41 参数，总25,661 |
| 目标 Dev | 原始 Clean/10–40 dB，额外合成 5/0 dB；88 条记录/每条件2474周期 |
| 原固定对照 | A′+LP3、A′+LP4，三个种子的现有预测文件 |
| Final-Holdout | 严禁触碰347条，直到最终方法与评估协议定稿 |

训练随机流直接复用 V2/V4 的 v2_experiment.source_data、raw_cycles、fresh_noisy_source 和 fixed_noisy_validation。V5 只训练新增 nahsf；绝不重新训练旧 LP3/LP4 权重，也不修改原成绩。

对照文件：

- Clean、10–40 dB：records/v4_hpasf/server_b4801df/results/evaluations
- 合成5、0 dB：records/v4_strong_snr/server_84f6757/results/evaluations

summarize 会严格核对每份 JSON 的 seed、method、condition、88 个记录、2474 个周期、truth 序列长度及相同顺序；输出对两个强对照的逐种子 Macro-F1 配对差值和两模型各自正确的窗口数。

## 4. 必须保存的门控和类别诊断

- 每训练轮：源 Val Clean F1、10/20/30 dB F1、联合 CE、门控最后一层 bias 梯度绝对值均值、门控 bias。
- 每个实测测试条件：Macro-F1、Accuracy、混淆矩阵、每类 F1/Recall、健康误报和故障误判正常。
- 每个实测条件：α mean/std/min/max/P10/P50/P90，基于真实类别的 α 均值，以及三个无标签描述符的均值。
- 目标 Dev 上的真实类别只允许在**测试分析结果生成阶段**分组；不得作为训练或早停输入。
- 重点看 0、5、10 dB 的三 seed 配对改进、OUS(A)/OLS(A) 类别和门控 α 是否存在可解释的变化。若 α 不随噪声变化，不能声称“噪声感知”已被学习；若只超过 LP3 没有超过 LP4，也不能声称取得最佳结果。
- 强噪声 0/5 dB 为 Clean 与原10 dB配对噪声缩放生成的 synthetic stress conditions，不是 HUST 作者公布的真实0/5 dB采样。

## 5. 服务器执行顺序

从仓库根目录执行（Python环境应复用旧 V4 环境）：

    git fetch origin
    git switch research/na-hsf-v5

    python scripts/v5_nahsf.py
    python scripts/v5_experiment.py smoke
    python -m pytest -q tests/test_v5_nahsf.py tests/test_v5_experiment.py

    # 三个随机种子，训练 only V5
    python scripts/v5_experiment.py train

    # 原始 MAT SNR 数据，参数须指向含 SNR_10...SNR_40 文件夹的父目录
    python scripts/v5_experiment.py evaluate --snr-root "/path/to/original_SNR_parent"

    # 合成压力测试，指向含 SNR_0 和 SNR_5 文件夹的父目录
    python scripts/v5_experiment.py evaluate --strong-snr-root "/path/to/generated/target" --conditions 5 0

    # 汇总并与原 V4 的 LP3/LP4 配对
    python scripts/v5_experiment.py summarize --conditions clean 10 15 20 25 30 35 40 5 0

默认输出 outputs/v5_nahsf/，拒绝覆盖已有 checkpoint 和同条件 eval；若只跑原始 SNR，可直接用 summarize 的默认条件。

服务器在完整训练前必须先跑测试并核对 CUDA 确定性、首次 MLP 梯度是否有限且非零、CNN 初值是否与旧种子一致。同步 GitHub 时推荐只提交 run.json、history.json、SUMMARY.json、PAIRED_V4_COMPARISON.csv 等诊断，不提交私有环境或大体积原始电流。

## 6. 结论边界

V5 代码完整可运行不代表诊断效果一定优于固定低通。新的研究假设是：可学习、逐输入变化的第四谐波增益，可通过引入当前信号质量特征在降噪与故障信息保留之间实现更适当的选择。实验必须与原先已经表现更好的 LP4 直接对比，不得隐去更强基线；后续论文创新点和独立 Final-Holdout 结果应在确认真实增益后再决定。
