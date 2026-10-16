# NCPHR-Net 会议论文：问题驱动的写作主线、方法贡献与消融实验设计 V1

> **状态：论文写作与实验设计方案，2026-10-16；并非已完成的论文，也不代表待测消融已取得结果。**
>
> **决定**：现有 B1 作为本文的提出算法，不修改 B1 全模型结构/损失；论文暂命名 **NCPHR-Net (Noise-Calibrated Phase-Aware Harmonic Recovery Network)**。本文件用于冻结叙述逻辑、确认每项方法贡献对应的可检验实验。正式实验写作暂按用户选择的 Seed 42、44 两次运行整理现有结果；这些是已完成实验的特定种子子集，不能称作全部已运行种子的统计结果。
>
> **写作硬约束**：在 **Section III: Experiments** 之前，Abstract、Introduction、Proposed Method **不得出现 HUST 数据集名称、具体五相 IM/六相 PMSM 实验平台、实际文件与样本数量、六分类标签编码及专属工况**。论文在 I–II 章的对象始终是 **multiphase motor drive systems** 中的 **inverter open-switch fault diagnosis**，以一般的 \(M\) 相电流建模。技术主线由工程问题推出，不从已有数据表倒推。真正的具体实验设置在 III-A 首次出现。

---

# 1. 一句话明确论文究竟研究什么

**研究对象**：多相电机驱动系统（multiphase motor drives），重点是驱动逆变器（power converter/inverter）的开关器件开路故障，而非轴承或电机绕组故障。

**应用动机**：更多电飞机（more-electric aircraft, MEA）、航空机电作动器（aircraft electromechanical actuators, EMA）、电推进及其他高可靠电驱动系统需要容错诊断。航空属于**代表性应用背景**，不宣称本文数据来自航空平台，更不把“航电计算机/导航通信电路板”与电机驱动逆变器混为一谈。航空应用引言的事实性陈述应由真实航空电驱动文献支持。

**研究挑战**：真实故障标签难以系统获取，仿真域可以提供训练数据；当部署到真实测量环境且电流受强噪声污染时，开路故障相关的谐波/相间电流差异容易被噪声掩盖。单纯固定截断高阶谐波无法继续处理保留低阶谐波内部的残余噪声；单纯逐相归一化又不能充分表达不同相之间的电流强弱差异。

**本文解决思路**：利用**噪声校准、跨相谐波描述符和受约束的连续复谱收缩**恢复低阶电流，再通过**波形形状 + 共享标度相间幅值的双视图**分类；在**纯仿真域**使用干净/含噪配对训练，并加入复谱、相间幅值和噪声校准恢复监督。

**不主张**：新发明 FFT、ResNet 或 Wiener-like gain；不声称提出通用 domain adaptation / 任意相数可直接运行的模型；不声称推理时获得真实噪声方差或无误差相位重建；不把具体测试集设计当作算法动机。

# 2. Introduction 的漏斗逻辑（六段）

| 段落 | 要回答的问题 | 写作内容 | 与下段的衔接 |
|---|---|---|---|
| P1 应用背景 | 为什么研究高可靠电机驱动？ | 更多电飞机/航空电动执行系统中的电驱动可靠性需求；多相驱动的容错潜力，避免航空专用能力的夸大 | 容错不代表无需检测故障 |
| P2 故障诊断 | 多相驱动有哪些关键功率电子故障？ | 逆变器开关开路会产生电流畸变、相间失衡并影响安全运行；需要识别 | 数据驱动方法的作用 |
| P3 训练困难 | 为什么选用仿真域训练？ | 深度学习依赖标注；真实故障采集成本高且破坏性强；仿真训练 + 实测推理具有现实价值 | 实测环境还有噪声和域差异 |
| P4 强噪声挑战 | 为什么一般神经网络不够？ | 噪声会掩盖电流波形畸变与谐波差异；高噪声条件使跨域诊断更困难 | 需要噪声约束 |
| P5 技术缺口 | 为什么固定滤波不够？ | 低阶谐波蕴含开路故障信息，固定谐波滤波消除带外噪声但保留带内干扰；独立归一化可能弱化相间电流强度证据 | 引出噪声条件化恢复 + 故障证据保护 |
| P6 本文方法 | 提出什么、如何解决？ | 概述 NCPHR-Net，两项设计及三条贡献 | Section II 技术实现 |

**约束**：不在引言 P4/P5 编造“已有文献均忽略某问题”的绝对断言；最终引用已核实的 sim-to-real 多相驱动诊断、噪声鲁棒诊断、频域滤波/可学习频谱收缩等实证文献。Introduction 最后不列具体平台、类别或数据数量。

## 2.1 Introduction 最末段建议的英文草稿（需配引文）

> To address these challenges, this paper proposes a **Noise-Calibrated Phase-Aware Harmonic Recovery Network (NCPHR-Net)** for noise-robust open-switch fault diagnosis in multiphase motor drive systems. Rather than relying solely on fixed harmonic truncation, the proposed framework estimates input-dependent spectral noise and exploits inter-phase harmonic relationships to perform bounded, phase-preserving shrinkage of low-order complex harmonic coefficients. A complementary dual-view representation then integrates individual phase-current waveform patterns with shared-scale inter-phase amplitude information. The network is optimized using paired clean and noise-contaminated simulated currents, with auxiliary spectral-recovery and fault-structure-preserving objectives, and performs diagnosis on measured currents without requiring target-domain labels for training.

这里的 phase-preserving 严格是 **乘以非负实数增益而不额外旋转观测复相量**，不能解释成“恢复真实相位”或“完全不改变时序波形”。“fault-structure-preserving”主要体现在相间强度损失及双视图，仍需消融验证增益。

## 2.2 末段 Contributions：建议固定两项算法贡献 + 一项实证贡献

**Contribution 1 — Noise-calibrated, phase-aware harmonic recovery**

利用保留频带外的谱功率统计构建输入相关噪声代理，进行可学习的逐谐波阶次功率校准；融合各相低阶谐波功率、相间相对功率和复谱 Gram 一致性，生成非放大、保留观测复相位方向的自适应谱系数增益，缓解低阶频带内残余噪声问题。

**Contribution 2 — Fault-aware dual-view diagnosis with structure-preserving supervision**

针对多相开路故障同时体现波形畸变与相间电流强度差异的特性，将逐相 MD 波形与全相共享基波尺度波形组成双视图；使用干净/含噪仿真配对学习，并通过复谐波重构、相间相对谐波强度和噪声功率校准监督促进故障相关结构保留。

**Contribution 3 — Experimental evidence**

通过纯仿真训练、真实测量测试的强噪声开路故障诊断，比较通用 CNN、相关文献复现、固定低阶谐波基线及学习式谐波恢复方案，并开展**与贡献 1/2 直接对应的消融**。此项是实验贡献，不虚构第三个独立网络模块。

> 两项贡献是**拟证明的技术主张**，不是提前宣布各个组件已经被严格消融证明有效。正式稿根据实测消融结论保留、调整或弱化对应机制表述。

# 3. Section II Proposed Method（仅用通用 \(M\) 相符号）

建议篇幅顺序：II-A Problem Formulation（短） → II-B Noise-Calibrated Phase-Aware Harmonic Recovery（最多） → II-C Dual-View Fault Representation → II-D Joint Recovery and Classification Optimization。

## II-A. Problem Formulation

- 定义仿真有标签训练集合 \(\mathcal D_s=\{(x_i^s,y_i^s)\}\)，\(x_i^s\in\mathbb R^{M\times N}\) 表示一个对齐电气周期的多相电流；部署域测量为 \(x^t=s^t+n^t\)。
- 目标：\(\hat y=f_\theta(x^t)\)，权重仅从源域学习并由源域验证选择；不使用实测故障标签参与模型训练与 checkpoint 选择。
- 说明完整电气周期、整数谐波阶次与相间同步测量的作用；模型适用于满足这些采样前提的多相驱动。
- 数学框架以 \(M\) 表示相数，具体实现中部分统计/卷积层适配目标相数；**不要凭一般数学符号宣称现有权重无需改造就可跨不同相数通用。**

## II-B. Noise-Calibrated Phase-Aware Harmonic Recovery（核心）

**B1 实际代码依据：** [\`scripts/hust_b1_phase_harmonic.py\`](../scripts/hust_b1_phase_harmonic.py)。

1. **频谱与共享尺度**：
   \[
   Y_{p,k}=\operatorname{rFFT}(x_p)[k],\quad
   s=\max\{\operatorname{median}_{p} |Y_{p,1}|,\epsilon_s\},\quad
   Z_{p,k}=Y_{p,k}/s.
   \]
   在保留低阶谐波中做后续恢复，避免每相独立幅值归一化抹去弱相差异。

2. **高阶带内统计估计噪声代理**：
   \[
   \rho_p=\operatorname{median}_{k\in\mathcal K_n}|Y_{p,k}|^2/s^2,\qquad
   \widehat v_{p,k}=\max\{\rho_p\exp(a_k),\epsilon_v\}.
   \]
   \(\mathcal K_n\) 为预定义的高阶分析频带（具体阶次在实现配置或实验设置中给出）；\(a_k\) 是可学习阶次校准参数。它是**噪声功率代理**，不是物理上严格精确的噪声 PSD。

3. **多相关系描述符**：由 \(Z_{p,k}\) 提取功率 \(|Z_{p,k}|^2\)、相间中位数归一化功率、同阶复谱 Gram 的实部归一化相干、噪声代理、对数信噪强度指标及谐波阶次；1D Conv 对**谐波阶次轴**处理这些描述符，生成有界修正 \(\Delta_{p,k}\)。不要称作另一个跨模态注意力或 VSD 模块。

4. **非放大谱系数收缩**：
   \[
   g_{p,k}=\sigma\left[
   \log\frac{|Z_{p,k}|^2+\epsilon}{\widehat v_{p,k}+\epsilon}
   +2\tanh(h_\theta(F)_{p,k})
   \right],\qquad
   \widehat S_{p,k}=g_{p,k}Y_{p,k}.
   \]
   \(0<g_{p,k}<1\)。任意额外复数残差都不被生成；它保留了**观测复谐波方向**，但不能修复噪声造成的复相位偏转。带外系数置零后用 iFFT 恢复多相电流。

**这一节建议的图**：频谱分解 → 噪声代理 → 跨相谐波描述 → 调整器/噪声校准 → 受约束 Gain → iFFT。图中显式标注复谱与实值权重不同。

## II-C. Fault-Aware Dual-View Representation

- **Waveform-shape view** \(Z_{\rm shape}=\mathrm{MD}(\widehat x)\)：对各相分别中心化与幅值归一，突出波形畸变。注意 MD 对低电流相包含特殊分支，不能绝对声称“完全丢失幅值”。
- **Shared-scale amplitude view** \(Z_{\rm amp}=\mathrm{Resample}(\widehat x/a(\widehat x))\)，其中 \(a\) 由全部相共享的**基波幅值中位数**换算得到；能更直接地保留相间相对电流强度、弱相/开相信息。
- \[
Z=\operatorname{Concat}(Z_{\rm shape},Z_{\rm amp}),\quad
\hat y=\operatorname{Classifier}(\operatorname{ResNet1D}(Z)).
\]
- 两种表示来自**同一批多相同步测量**，不能称作不同传感器或不同物理模态。
- 强调“不同诊断线索的互补”是设计动机；只有消融支持后才能写成经验证的效果结论。

## II-D. Joint Recovery and Classification Optimization

从仿真源域构造干净 \(x_s\) 与注入噪声后的 \(x_s+n_s\) 配对：

\[
L_{\mathrm{cls}}=\tfrac12\mathrm{CE}(f_\theta(x_s),y)+
\tfrac12\mathrm{CE}(f_\theta(x_s+n_s),y).
\]

现有 B1 使用：

\[
L_{\mathrm{rec}}=L_{\mathrm{complex}}+
0.4L_{\mathrm{strength}}+0.1L_{\mathrm{noise}},
\qquad L=L_{\mathrm{cls}}+1.0L_{\mathrm{rec}}.
\]

- **Complex**：恢复的低阶复谱与干净源复谱的实部、虚部归一化 SmoothL1；
- **Strength**：第 1–4 阶复谱幅值合成的每相能量相对于相间中位数的对数 SmoothL1；约束弱相/相间相对强度，**不是角度相位差损失**；
- **Noise**：模型预测低阶噪声功率与**源训练已知注入噪声功率**的 batch×phase 均值，在 log 空间的 SmoothL1。
- 先在仿真域开展 10 个 epoch 仅恢复器预训练，再以配对分类损失和恢复损失联合训练 50 个 epoch。
- 清晰区分训练（可访问**源域** Clean 真值及注入噪声）与推理（仅访问**目标域**含噪观测）。

# 4. Section III Experimental Results：**首次出现具体数据集、平台和设置**

## III-A. Experimental Setup（本节首次介绍实际数据）

- **实验平台**：HUST 多相驱动开路故障数据，当前正式完整结果来自**五相感应电机 (five-phase IM) 驱动系统**；六相 PMSM 如后续完成独立验证，再在这里介绍，未完成不写成已有实测贡献。
- **六类标注**：Healthy、OUS(A)、OLS(A)、OP(A)、OUS(A)+OUS(B)、OLS(A)+OP(C)。
- **仿真源域**：63 个 MAT 文件、13 工况；Source-Cycle V2：962 Train 周期、221 Val 周期、78 Gap 周期。单周期五相同步电流，重采样每相 4096 点。
- **实测目标域**：88 条记录家族 Target-Dev，每个 SNR 条件 2474 周期；另有未使用的最终 holdout，待最终算法和协议确定后再评价。
- **训练**：仅仿真数据；每轮 Clean 与从 10/15/20/25/30/35/40 dB 中抽取的随机噪声视图配对；源 Val 使用 Clean 和 0/5/10/20/30 dB，先 Macro-F1 后 CE 的源域独立选模；Adam、lr=0.001、wd=0、batch=32、联合50轮、恢复器预训10轮。
- **实测噪声来源**：5、0、-5、-10 dB 是基于作者给出的 10 dB 实测残差缩放构造的**压力测试条件**，不是四批独立采集的测量结果；要在噪声协议中准确写清生成公式。
- **统计口径**：当前准备的正式工作表按已指定 Seed 42、44 的均值±**样本**标准差整理，并标注 n=2；不称其为全部历史运行的统计结果。最终是否保留该口径，在文章定稿时保持真实一致。
- **指标**：Macro-F1 主指标；Accuracy、每类 Recall、典型混淆矩阵辅助。所有模型与消融必须执行同一 Source-Cycle V2、同一源训练及选模规则。

## III-B. Comparison with Existing Methods（6 baselines + proposed）

| 方法 | 作用 | 当前 Source-Cycle V2 状态 |
|---|---|---|
| Plain CNN1D | 通用无残差卷积分类器 | 已完成 |
| ResNet1D | 通用残差分类器，未引入显式谐波处理 | 已完成 |
| Xu-Full（根据原论文自行复现 CAE+DML） | 与相关多相驱动故障诊断方法对比；注明非官方公开权重 | 已完成 |
| ResNet+LP3 | 固定保留 DC+前三阶谐波 | 已完成 |
| ResNet+LP4 | 固定保留 DC+前四阶谐波；重要强基线 | 已完成 |
| B0 | 历史学习式谐波恢复方案，可展示结构代际变化，但**不能用它严格归因 B1 的单一模块** | 已完成 |
| **NCPHR-Net (=B1)** | 本文提出方法 | 已完成 |

**正式对比表计划：** 实测 5/0/-5/-10 dB 的 Macro-F1（均值±样本标准差）；Clean 可作补充控制，不作为论文主卖点。画与表一致的噪声退化曲线，标注真实/派生噪声条件。

当前 Seed 42/44 子集的已有 B1 指标（%）：5 dB **98.81±1.25**；0 dB **97.46±0.54**；-5 dB **91.80±0.53**；-10 dB **71.88±0.61**。这些不能提前填入消融模型行。

## III-C. Ablation Study：从“论文主张”严格反推实验

### 4.1 消融不是网络零件清单

消融的目标是回答**如果删去/替换一个被宣称有用的技术机制，诊断性能如何变化**。每个消融事先写明假设、唯一修改因素、保持不变条件、可以支持的结论和不能支持的结论。若无提升，不强行将相应模块定性为主要贡献。

### 4.2 优先核心消融组（建议新增 5 个可复现模型配置）

| ID | 对应论文主张 | 唯一主要改动 | 与 Full B1 比较后回答什么 |
|---|---|---|---|
| **A1 Fixed LP4 + Dual-View** | 贡献1：学习式恢复单元的总体价值 | 将学习式谐波恢复器替换成固定 k0..4 投影，保留 MD+共享尺度双视图和同 ResNet 分类器；该前端不需要恢复预训/辅助损失 | 自适应恢复单元**作为整体**是否带来增益；**不能**独立归因噪声代理、Gram 或辅助监督 |
| **A2 w/o Sample-wise Noise Proxy** | 贡献1：输入相关噪声校准 | 在恢复器中以**仅用 Source-Train 预计算的常数噪声代理**替换每样本每相的 k20..80 噪声代理；同一路径喂给 variance 与 adjustment，其他模型和损失保持不变 | 当前样本级噪声可靠性是否优于固定噪声先验；输入信号能量仍会变化，所以不能称为完全“无噪声条件化” |
| **A3 w/o Inter-phase Recovery Descriptors** | 贡献1：跨相谐波描述符 | 仅在 adjustment 的输入特征中，将 **relative_power** 与 **Gram phase_consistency** 两组通道归零或按严格等参方式屏蔽；保留信号功率、噪声代理、logSNR、阶次，保留共享基波尺度/双视图/strength loss | **恢复增益网络中**跨相相对功率及同阶一致性描述符是否有益；**不能**声称完全移除了模型一切相间信息 |
| **A4 MD-only** | 贡献2：双视图的共享幅值分支 | 恢复器/损失/训练阶段不变；分类头只输入 MD 五相视图，stem 通道数相应改变 | 加入共享尺度幅值视图是否在 MD 之外贡献额外信息 |
| **A5 w/o Strength Loss** | 贡献2：相间强度结构监督 | 保持完整恢复器+双视图+预训10/联合50轮；仅将 phase-strength loss 权重由0.4设为0，其他复谱/噪声损失不变 | **额外相间强度监督**是否产生分类收益，而不是仅降低辅助 loss |

#### A1 的因果边界

Full B1 带有学习式增益、10轮恢复预训练及辅助恢复损失；A1 用固定 LP4 时这些步骤均无意义。因此 A1 是**完整自适应恢复子系统的组合消融**，不是“只把 Gain 改成 1”的纯单变量验证。若要进一步单独归因学习 Gain，应另设相同网络但固定 Gain 的结构对照；会议论文首轮不必强行增加。

#### A2 的具体实现边界

- 定义 \(\rho_{\rm fixed}\) 为 Source-Train 的配对增强视图（按冻结噪声分档或指定样本子集）计算出来的、跨样本与相的稳健标量，仅在训练前离线固定，记载用于计算它的源数据范围和随机规则。
- 用 \(\rho_{\rm fixed}\) 代替原每周期每相 \(\rho_p(x)\)，同时替换计算 \(\hat v\) 和输入 Conv 的 proxy 特征；**不能只在推理阶段屏蔽，否则形成训练/推理错配**。
- 允许原有 \(a_k\) 继续学习，所有恢复损失包括 noise calibration 保持，因此测试的是“样本自适应代理”贡献，不是“无噪声估计”。
- 若观察到无法有效拟合不同训练噪声水平，这也是固定代理方案在研究目标上的机制局限，而不是代码错误。

#### A3 的具体实现边界

现有 B1 adjustment 特征按通道依次包含：5个 log1p(signal_power)，5个 log1p(relative_power)，5个 Gram coherence，5个 log1p(proxy)，5个 tanh(logSNR/8)，1个谐波阶次，总计 26 通道。A3 将**第2和第3组**屏蔽为零，其他全部不变；形状和 Conv1d 参数量保持一致。不能将它夸大成完全撤销共享尺度、双视图等所有相间信息。

#### A4 的具体实现边界

完整 B1 拼接两个五相视图得到10通道；A4 只保留5通道 MD，按此重建 ResNet stem 并**从头独立训练**。若 Full 显著优于 A4，支持共享幅值分支有增量贡献；但要证明“两种视图真正互补”，更有说服力的是再做 **Amplitude-only**（可选 A6）。

#### A5 的具体实现边界

取消 strength loss 不意味着“失去所有相间信息”——B1 的共标度、Gram、双视图仍在。A5 仅证明该**辅助恢复损失独立是否值得保留**。

### 4.3 可选补充组（有意义时再做，不必与首轮绑定）

| ID | 何时需要 | 具体操作 | 证明边界 |
|---|---|---|---|
| **A6 Amplitude-only** | 如果 Introduction 坚持“两个视图互补” | 只保留共享尺度五相视图，分类器独立训练；与 MD-only、Full 对照 | 双视图是否超过任一单视图 |
| **A7 w/o Recovery Supervision** | 如果 II-D 把配对恢复监督列为重要创新 | 保留 B1 全部前向结构，取消 10 轮恢复预训和全部恢复损失，只用配对 CE 训练50轮 | 检验**整套恢复监督训练策略**作用，不能归因其中某个 loss |
| **A8 w/o Noise Calibration Loss** | 如果要检验该损失项 | 只将 noise_cal loss 权重从0.1设0，保留动态噪声代理和其他结构 | 仅支持噪声校准**监督项**的贡献，不代表噪声代理无效 |

不建议一开始做所谓“去掉 CNN/去掉 FFT/去掉所有损失”等不明确变量的消融，也不需要为了增加表格行数而新增多小波、Transformer 或注意力。

### 4.4 消融实验的统一执行要求

- 采用 **Source-Cycle V2**，同一源域 Train/Val 清单与标签；同一 42/44 两种子；严格使用同一 Clean/Noisy 生成、batch32、Adam lr1e-3、wd0、50轮训练和源域选模规则。
- 对包含可学习恢复器+恢复监督的 A2/A3/A4/A5，继续源域 10 轮恢复预训；A1 无恢复器，A7 无恢复监督，无预训，并在表格脚注准确说明。
- **所有消融从头训练**；不能拿 Full B1 权重临时删模块后直接推理充当正式消融。
- 不因目标域 Dev 指标高低为某个消融单独选“最优 epoch”；模型 checkpoint 只依据源验证选择。
- 推理输入条件一致，5/0/-5/-10四档，每种子同时保存整体 Macro-F1、Acc、各类 Recall、预测记录及用于复核的权重哈希。
- 同时记录参数量与单周期推理延迟（若有测量资源），以确认改动的计算成本；不是每项消融都必须做独立延迟实验。
- 消融比较优先用 **相同种子逐种子配对差值**，然后再报告均值；以免均值掩盖单一随机初始化异常。
- 优先检验 -5/-10 强噪声提升，兼顾5/0以避免较高SNR明显退化。
- **未跑之前一律填写 T.B.D.**；根据实际消融结果决定能否成立“噪声校准有效”“跨相描述有效”“双视图互补”“结构损失有效”等宣称。

### 4.5 预期论文消融表（不是预填结论）

| 方法 | 自适应复谱恢复 | 样本级噪声代理 | 恢复增益的跨相描述符 | 双视图 | 强度保持loss | 5 dB | 0 dB | -5 dB | -10 dB |
|---|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| A1 Fixed LP4+Dual-View | — | — | — | ✓ | — | TBD | TBD | TBD | TBD |
| A2 w/o Sample-wise Noise Proxy | ✓ | — | ✓ | ✓ | ✓ | TBD | TBD | TBD | TBD |
| A3 w/o Inter-phase Recovery Descriptors | ✓ | ✓ | — | ✓ | ✓ | TBD | TBD | TBD | TBD |
| A4 MD-only | ✓ | ✓ | ✓ | — | ✓ | TBD | TBD | TBD | TBD |
| A5 w/o Strength Loss | ✓ | ✓ | ✓ | ✓ | — | TBD | TBD | TBD | TBD |
| **Full NCPHR-Net** | ✓ | ✓ | ✓ | ✓ | ✓ | 98.81 | 97.46 | 91.80 | 71.88 |

表中 Full 行是已归档 Seed42/44 两次运行的均值，A1–A5 是**计划重训**的模型。A1 包含训练预算差异，表中简化符号不能替代其定义文字；A4的“—”表示没有**双**视图，而不是没有输入视图。

## III-D. Diagnostic and Mechanism Analysis

优先 3 张图：
1. **框架图**（放 Section II）：清楚呈现真实推理路径与仅训练的辅助损失路径；
2. **四档噪声 Macro-F1 曲线**（III-B）：强调固定 LP3/LP4 vs NCPHR-Net 的噪声退化；
3. **代表性的实际电流/复谐波恢复图**（III-D）：干净参考、含噪观测、恢复波形及噪声功率/增益；图像须由真实实验运行数据与选中模型产生，不能用理想示意冒充。若篇幅允许，附极低SNR混淆矩阵。

若进一步验证**不同多相驱动类型**的适用性，独立补充另一平台实验，前提是代码按相数改造并运行真实对照，不能提前把该项写为已完成贡献。

# 5. 写作与实验一一对应的闭环

| Introduction 末段主张 | Methodology 公式/模块 | 关键消融 | 证据不足时如何调整文字 |
|---|---|---|---|
| 固定频域截断后仍有带内噪声，需要适应性收缩 | II-B \(\hat S=gY\) | A1 vs Full | 若无改善，弱化自适应恢复的性能优越性主张 |
| 输入相关的噪声校准提高收缩决策适应性 | II-B \(\rho_p,\hat v_{p,k}\) | A2 vs Full | 若无改善，只保留为设计实现，不单独列贡献 |
| 跨相复谱关系帮助判别可靠谐波 | II-B \(F_{\rm gram},F_{\rm relative}\) | A3 vs Full | 若无改善，不强调跨相调节器为必要机制 |
| 波形形状与相间幅值提供互补信息 | II-C \(Z_{\rm shape},Z_{\rm amp}\) | A4 vs Full；加强证据时A6 | 若缺少A6，措辞为共享幅值分支的增量作用，而不是完全证明双向互补 |
| 去噪过程中保护故障相间强度 | II-D \(L_{\rm strength}\) | A5 vs Full | 若无改善，剔除“结构损失独立有效”之说 |

# 6. 写作推进顺序（冻结方法，按证据完善叙述）

1. 固定 Introduction 的逻辑与两项算法贡献，检查航空电驱动背景及现有方法引用的可信度。
2. 按真实代码绘制总架构图，撰写 II-B/II-C/II-D，校验每个数学符号的尺度与代码一致。
3. 同时执行 **A1–A5（Seed42/44）** 的五配置消融；每项保留源域训练记录、选中权重与目标实测逐周期预测。
4. 输出主性能表与四档曲线；单独输出五行消融表，再决定是否补做 A6–A8。
5. 用真实数据绘制恢复图、分析类间特征保留，对照消融结果适当修正 Introduction 中两项技术贡献。
6. 在算法/训练配置冻结后完成约定的最终实验与论文定稿；不以临时挑选的目标域 checkpoint 代替源域选择。

## 现有仓库依据

- B1 完整前向与损失：[scripts/hust_b1_phase_harmonic.py](../scripts/hust_b1_phase_harmonic.py)
- 新源域训练/实测测试统一基准：[docs/SOURCE_CYCLE_BENCHMARK_V1.md](SOURCE_CYCLE_BENCHMARK_V1.md)
- 冻结实验配置：[configs/source_cycle_benchmark_v1.json](../configs/source_cycle_benchmark_v1.json)
- 实验结果与证据：[records/source_cycle_benchmark/server_4728e9b/SERVER_RESULTS.md](../records/source_cycle_benchmark/server_4728e9b/SERVER_RESULTS.md)
- 先前原型设计与公式：[docs/HUST_A1_B1_V1.md](HUST_A1_B1_V1.md)

**本文档只保存论文故事与消融计划，不修改代码、不启动训练、不覆写既有实验数据。**
