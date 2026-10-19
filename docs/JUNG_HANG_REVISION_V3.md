# Jung / Hang V3：输入机制修正与训练协议

> 2026-10-19：42/44四组50轮训练、28组七档Dev测试及预测回放已完成，见[V3客观实验记录](JUNG_HANG_CYCLE_V3_RESULTS.md)。运行前修复了一处noise_records模块导入错误；网络和冻结配置未修改。

## 状态与边界

V3 是独立于 V2 的新版本。原 V2 代码快照、配置、权重和已发布结果不修改。V3 的目标是修正已识别的算法偏离，随后重新训练与评估；代码更新本身不代表性能已经提升。

两篇原论文没有公开完整训练代码及全部层级超参数，因此本版本是**论文机制指导下的五相任务适配**，不是作者代码的逐参数复现。特别是三相 Clarke 轨迹到五相 VSD 轨迹的迁移，属于任务适配。

## 1. Jung Modified CNN

### 保留的网络主干

- 一维扩张卷积；
- BatchNorm 与 ReLU；
- 最大池化；
- 两层卷积组成的残差块；
- Global Average Pooling 与六分类头。

这一主干已经与原文 Fig. 9(b) 描述的“dilated convolution + residual block + GAP”相符。原论文的 7,213 个参数对应原三相输入和原输出类别数，不能要求五相六分类适配后参数量完全相同。

### V3 输入修正

旧输入预处理对五个相电流分别缩放，可能抹除故障引起的相间幅值比例差异。V3 对每相做窗口中点去偏置，但五相共用一个尺度：

\[
x'_p(t)=\frac{x_p(t)-\frac{x_{p,\max}+x_{p,\min}}{2}}
{\max_{q,t}\left|x_q(t)-\frac{x_{q,\max}+x_{q,\min}}{2}\right|+\epsilon}.
\]

这样移除窗口的公共幅值尺度，同时保留相间相对幅值关系。输入仍是每周期 128 点、五通道信号。

### V3 噪声增强修正

原论文讨论了在电流被开关故障物理阻断的区间限制噪声注入，并使用 THD+N 定标；当前五相 IM-AA 标签不包含原 NPC 逆变器的开关拓扑，不能直接照搬原论文的开关特定限幅逻辑，也不能未经验证地把 NPC 电流的 3%/5%/7%/10% THD+N 设定迁移到电机电流。

因此 V3 保留当前 SNR 评估协议，训练时使用 10–40 dB 的逐记录 AWGN，并增加短时 RMS 静默区门控：
- 依据每相短时 RMS 识别持续低电流区；
- 这些区间不注入噪声，减少把故障断流区“填满”的风险；
- 对剩余噪声按每相整条记录的目标功率比重新标定。

这是一种通用的五相任务适配，**不等同于原论文的 THD+N 方法或其开关状态限幅器**。其效果仍须通过训练实验验证。

## 2. Hang WCNN

### V3 轨迹输入

旧版采用两个 VSD 子空间图、64×64 图像、窗口级共享尺度。它既不同于原论文的单个归一化 Clarke 轨迹，也没有实现原文的逐时刻矢量模长归一化。

V3 改为：
1. 五相电流按相去除窗口均值；
2. 通过五相 VSD 投影提取基波平面 \((\alpha_1,\beta_1)\)；
3. 对每个时刻单独归一化矢量模长：

\[
\hat{\boldsymbol i}_{\alpha\beta}(t)=
\frac{[\alpha_1(t),\beta_1(t)]^\top}
{\sqrt{\alpha_1(t)^2+\beta_1(t)^2}+\epsilon};
\]

4. 对近零矢量不做方向放大，跳过对应无效轨迹点；
5. 将轨迹栅格化为单通道 40×40 图像。

这保留原方法“固定半径轨迹突出形状差异”的机制，同时将三相 Clarke 投影替换为五相基波 VSD 投影。该替换是可解释的任务适配，但仍需确认数据集的相序与物理相轴定义。

### V3 WCNN 结构

- Conv2D 通道：4 → 8 → 16 → 32 → 64；
- 每个卷积阶段后使用 BatchNorm 和 ReLU；
- 在第 1、2 个卷积阶段后使用固定 Haar LL 池化；
- 40×40 → 20×20 → 10×10；
- 最终展平并接六分类线性层。

这对应原文 Fig. 6 及正文给出的五个卷积阶段和两次 DWT pooling。Haar LL 核采用原文式（10）–（12）的未归一化 2×2 全 1 滤波器；高频子带不传给后续层。

### V3 训练视图

Hang 原论文通过 DWT 池化处理噪声图像，并未提出与 Jung 相同的成对 Clean/Noisy 交叉熵训练。因此 V3 默认仅用 Clean 源域样本训练 Hang；源域带噪验证仍用于统一的 source-only checkpoint 选择。若后续要研究配对噪声训练，应将其作为单独消融，不要混入主版本。

## 3. 实验协议

- Source-Cycle V2 划分不变：962 Train、221 Val；
- 种子 42、44；50 epochs；Adam，学习率 0.001；batch size 32；
- 训练阶段不读取目标域数据；
- Source-Val 的 Clean 与 0/5/10/20/30 dB 噪声结果用于选模；
- 目标 Dev 使用既有固定 88 条记录、每档 2,474 个周期；
- 不评估 347 条 Final-Holdout；
- V3 输出使用独立目录，不能覆盖 V2 的权重、预测和报告。

## 4. 代码文件

- \`scripts/jung_hang_models_v3.py\`：V3 网络、输入适配器和静默区门控噪声；
- \`scripts/compare_jung_hang_v3.py\`：模型工厂、原始窗口转输入、合成 smoke test；
- \`scripts/jung_hang_cycle_v3.py\`：冻结周期划分训练、Dev 评估、预测回放和汇总；
- \`configs/jung_hang_cycle_v3_v1.json\`：冻结 V3 配置；
- \`tests/test_jung_hang_models_v3.py\`：CPU 单元测试。

## 5. 先测试，再正式训练

在仓库根目录运行：

\`\`\`bash
python scripts/compare_jung_hang_v3.py
pytest -q tests/test_jung_hang_models_v3.py
\`\`\`

V3 周期运行器需要服务器上的冻结源数据、参考代码快照、附加 SNR 数据和上一轮十二模型结果。先确认这些路径与服务器现有目录一致，然后使用独立输出目录，例如：

\`\`\`bash
python scripts/jung_hang_cycle_v3.py train --method jung --seed 42 \\
  --reference-root /PATH/TO/REFERENCE_OUTPUT \\
  --source-root /PATH/TO/SOURCE_DATA \\
  --additional-root /PATH/TO/ADDITIONAL_DATA \\
  --previous-output /PATH/TO/PREVIOUS_12_MODEL_OUTPUT \\
  --output /PATH/TO/NEW_V3_OUTPUT
\`\`\`

对 \`jung/hang × seed 42/44\` 分别运行训练，再运行 \`evaluate\`、\`verify\` 和 \`summarize\`。不得将新结果写入旧 V2 记录目录。

## 6. 解释结果时的限制

1. V3 的输入和训练协议改变了，所以不能把 V3 的分数当作只改一行网络层的消融结果。
2. 如果性能改善，仍需至少三个种子复核；目前冻结配置中的两个种子是延续此前实验的起点，不足以形成稳定性结论。
3. 若 Hang 仍然失败，应继续检查实验域轨迹是否被逐时刻归一化后变成相似圆弧，以及近零矢量占比；不能直接把问题归咎于 DWT。
4. 不以目标 Dev 指标反复选择图像大小、归一化或噪声阈值。需要探索时，应在 Source-Val 完成选择，再冻结版本并对 Final-Holdout 进行一次独立评估。
