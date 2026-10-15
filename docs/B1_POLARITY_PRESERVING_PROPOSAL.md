# Candidate 2 — B1-Polarity：正负电流极性结构保持的谐波恢复

> **状态：设计规格（Design Proposal），尚未实现、训练或获得新指标。**
> 
> 基线代码：[`scripts/hust_b1_phase_harmonic.py`](../scripts/hust_b1_phase_harmonic.py)；源域划分：Source-Cycle V2，主要随机种子 **42/43/44**。
>
> 思想来源：IADC-Net（Gao et al., MSSP 2026）针对非对称故障特征设计偏移软阈值的**动机**，而不是将其具体 \(B_c\) 公式照搬至复谐波。这里的极性结构保持方法是面向**开关管开路电流**的独立研究假设。

## 1. 故障机理与具体假设

HUST 五相 IM-AA 数据的六类为：

0. Healthy
1. OUS(A)：A 相上管开路
2. OLS(A)：A 相下管开路
3. OP(A)：A 相开相
4. OUS(A)+OUS(B)：A、B 相上管开路
5. OLS(A)+OP(C)：A 相下管开路并 C 相开相

上下开关失效可能影响不同电流极性的导通路径，使对应相的正向/负向电流出现**不均衡**。但是不能无视续流路径、控制策略和电流方向定义，把“上管开路必然消失正半波”“下管开路必然消失负半波”写成未经验证的规则。

现有 B1 的 `phase_strength()` 通过第 1–4 阶复谐波的能量包络约束不同相之间的相对强度。这能保护 **哪一相接近开相/弱电流**，但该能量包络本身不包含 **某相正、负极性各占多少** 的完整信息。

B1 还有第 0–4 阶复谱重构损失和双视图 CNN，已经间接包含部分不对称性；因此本提案不是补充全新的观测数据，而是**重新分配训练约束的关注点**。能否提高诊断性能取决于当前 B1 是否真的在强噪声下破坏了这类特征。

## 2. “正负半周”应如何严格定义

**不要按采样索引**将一周期前 2048 点视为正半周、后 2048 点视为负半周；电角度起点未必对准零交点。

对于同一相电流 \(x_p[n]\)，定义**电流极性区域**：

\[
x_p^+[n]=\max(x_p[n],0),\qquad
x_p^-[n]=\max(-x_p[n],0).
\]

每极性的面积指标为（单位与原电流相同）：

\[
A_p^+(x)=\frac1N\sum_{n=0}^{N-1}x_p^+[n],\qquad
A_p^-(x)=\frac1N\sum_{n=0}^{N-1}x_p^-[n].
\]

候选不对称比值为：

\[
\eta_p(x)=\frac{A_p^+(x)-A_p^-(x)}
{A_p^+(x)+A_p^-(x)+\varepsilon}
=\frac{\operatorname{mean}(x_p)}{\operatorname{mean}(|x_p|)+\varepsilon}.
\]

在非零相的理想示意中：正、负电流面积接近时 \(\eta\simeq0\)，正极性占优时正，负极性占优时负。它是**诊断统计量**，不是对所有故障类别都一定有区分力的预定物理判据。

### 2.1 为什么不直接将 \(\eta\) 作为惟一损失？

开相与弱相的 \(A_p^++A_p^-\) 很小，少量噪声就可能使 \(\eta_p\) 的符号大幅变化。更稳妥的做法是保护 **两个正负面积本身**，而不是仅让一个不稳定的比值相等。

此外，\(\eta\) 的分子是均值/DC 信息：

\[
A_p^+-A_p^-=\operatorname{mean}(x_p)
=\frac{\operatorname{Re}X_p[0]}{\sqrt N}
\quad(\text{使用 }{\tt rFFT(norm="ortho")}).
\]

因此它与 B1 已有的 k0 复谱监督部分**存在信息重叠**。不能把单独监督 \(\eta\) 说成模型获得了全新的物理观测。

## 3. 应从哪个“干净波形”计算监督目标？关键设计选择

B1 **只保留 k0..k4**；其恢复波形来自 k>4 全置零后的逆 FFT。如果直接令恢复低带宽波形拟合**原始干净波形的全部正负面积**，可能把无法重建的高阶畸变强加给模型。

因此首版建议用 **与 B1 相同信息带宽** 的干净仿真参考：

\[
X^{\rm clean}_{p,k}=\operatorname{rFFT}(x^{\rm clean}_p)[k],\qquad
x^{\rm clean,LP4}
=\operatorname{irFFT}(X^{\rm clean}_{k=0..4},X_{k>4}=0).
\]

对比目标是：
- 预测重构周期 \(\widehat x=\operatorname{irFFT}(\widehat X_{k=0..4},0)\)；
- 同带宽干净参考 \(x^{\rm clean,LP4}\)。

这让新增极性损失与现有 k0..4 谐波恢复目标**在可达到的表达空间内一致**。

### 3.1 统一幅值尺度（不能对每相单独归一化）

从 **干净源域当前配对周期**计算一个共享幅值尺度，仅在训练 loss 中用作固定目标尺度：

\[
s_{\mathrm{cur}}
=
\max\left(
\frac2{\sqrt N}\;
\operatorname{median}_{p=1..5}|X^{\rm clean}_{p,1}|,
\varepsilon_s
\right).
\]

使用 `detach()` 固定该尺度，且所有相共享同一个尺度；不能让模型通过缩小输出波形、或把弱相分别放大来平凡地最小化约束。

**这不是推理泄漏**：干净参考仅存在于源域训练损失，B1 `forward(raw)` 在实测域始终只读取含噪输入。实测 Clean 可以用于单独的离线误差审计，但不能被送入模型预测或用于选模。

## 4. 最小可执行候选：只新增极性结构损失

对每个相分别比较恢复后的正、负面积：

\[
\mathcal L_{\mathrm{pol}}
=\frac1{2M}\sum_{p=1}^{M}
\left[
\operatorname{SmoothL1}
\left(\frac{A_p^+(\widehat x)}{s_{\mathrm{cur}}},
\frac{A_p^+(x^{\rm clean,LP4})}{s_{\mathrm{cur}}}\right)
+
\operatorname{SmoothL1}
\left(\frac{A_p^-(\widehat x)}{s_{\mathrm{cur}}},
\frac{A_p^-(x^{\rm clean,LP4})}{s_{\mathrm{cur}}}\right)
\right],
\]

并对 batch 平均。

保留现有 B1 三项恢复损失：

\[
\mathcal L_{\mathrm{B1-recovery}}
=\mathcal L_{\mathrm{complex}}
+0.4\mathcal L_{\mathrm{phase-strength}}
+0.1\mathcal L_{\mathrm{noise-calibration}}.
\]

候选训练损失：

\[
\boxed{\mathcal L_{\mathrm{B1-pol}}
=\tfrac12\mathrm{CE}(\mathrm{clean})
+\tfrac12\mathrm{CE}(\mathrm{noisy})
+\mathcal L_{\mathrm{B1-recovery}}
+\lambda_{\mathrm{pol}}\mathcal L_{\mathrm{pol}}}
\]

源域恢复器 10 轮预训练也可使用 \(\mathcal L_{\mathrm{B1-recovery}}+\lambda_{\mathrm{pol}}\mathcal L_{\mathrm{pol}}\)，之后 50 轮联合训练沿用上式。

- `lambda_pol` **首版暂定 0.1**，是设计起点而不是经实验验证的最优值；必须在源域梯度幅值审计中核对其实际影响，必要时只通过源域规则固定新的数值。不能依据 Target-Dev 准确率反复搜索。
- **关键公平性**：与 B1 的核心区别仅为新增 loss；完全不改 FFT、噪声代理、增益 `gain`、k0..4、iFFT、MD+共享幅值双视图 ResNet 及训练/验证噪声。
- **推理开销**：新增损失只用于训练，模型部署时计算图、网络参数量及正常前向计算量理论上与原 B1 完全相同。

## 5. 这个损失保护了什么？没有保护什么？

### 保护

- 允许区分**正、负电流极性的面积**，不只使用每相总谐波能量。
- 保留相间弱电流信息：当一相接近开相，正负两项都应接近低值，而不是因相独立 MD 归一化放大。
- 对真正存在极性不对称的上/下管开路，提供训练阶段的显式监督信号。
- 保持 B1 的“无需推理时故障标签、无需真实 SNR、无需 Clean 参考”条件。

### 局限

- 两个波形可以具有相同的正、负面积，但不同局部畸变位置；因此 \(\mathcal L_{\mathrm{pol}}\) 不能代表完整波形信息。
- B1 的第 0–4 阶复谱恢复本来包含 DC 与形状信息，极性面积损失可能只是冗余加权；即使损失下降，不能自动声称新的故障可分性增加。
- 如果 HUST 具体开关控制机制没有形成可迁移的极性规律，新增损失可能削弱现有 B1 性能。
- 已有复谱增益不能旋转错误相位，也不能重建原本缺失的高频特征；极性损失无法绕过这一信息边界。
- 只增加极性损失**不等于已经建立完整“Physics-Informed Network”**；是否可以把它称为物理结构约束，需要分析具体电流与故障标签之间的稳定关联。

## 6. 与 seed43 类别错误的关系（必须具体，不要强行解释）

新 Source-Cycle V2、B1、**seed43** 的实测 Target-Dev **Clean**，OUS(A) 共 307 周期：124 正确、150 被判为 OUS(A)+OUS(B)、33 被判为 Healthy。

这不是“上下管极性方向区分失败”的直接证据，因为 OUS(A) 和 OUS(A)+OUS(B) **共有 A 相上管开路**，它们的差别主要在 **B 相是否同时上管开路**。

因此，对 seed43 应特别检查：

1. 真实 OUS(A) 与 OUS(A)+OUS(B) 在 **B 相**的 \((A_B^+, A_B^-)\)、\(\eta_B\)、总体相电流强度、前四阶谐波上的区分度；
2. 恢复前后的 B 相正、负面积是否发生系统性偏差；
3. 真正的混淆是否已经在实测 Clean 发生（已确认），因此不能把问题全部归因于极强噪声；
4. 与种子 42/44 的同类恢复波形、增益及输出分类分布比较；
5. 如果正/负面积本身没有稳定区分力，**停止以 seed43 错误为理由推动极性损失**，转向多相分类边界或工况迁移原因调查。

只有这些检查支持极性结构损坏，才有针对性地给 B1 增加极性损失的依据。

## 7. P3 极性特征调查：低成本、可独立完成

P3 是**前置机制确认**，不是论文必需增加的一大轮网络实验：

- **数据**：源域 Clean 仿真周期 + 实测 Target-Dev Clean 周期；后者用于**离线解释**，不是训练/选模数据。
- **取样**：每条记录均保留记录 ID、工况、类别、五相顺序；样本按记录分组，不把相邻周期当成独立物理试验重复。
- **变量**：每相正负面积 \(A_p^\pm\)，共享归一化面积，\(\eta_p\)，B1 恢复前后变化，关注 A/B/C 相。
- **比较**：Healthy；OUS(A) vs OLS(A)（同相不同极性）；OUS(A) vs OUS(A)+OUS(B)（B 相增量故障）；OLS(A) vs OLS(A)+OP(C)（C 相新增开相）。
- **输出**：每类/记录分组统计与跨域对照，少量代表波形，不必构造巨型可视化仪表盘。
- **判断**：若源域与实测域对同一类的正负面积关系方向基本一致，且 B1 恢复误差明显影响这些关系，进入该 loss 实验。若相关规律不存在或 B1 已完全保护，不强行实现。

## 8. 单元测试与服务器实现计划

建议新建独立训练模块 `scripts/hust_b1_polarity_loss.py` 或封装新 `B1PolarityRecoveryNet`，但 **forward 完全等于 B1**；仅扩展恢复目标。当前文件是方法方案，不是声称代码已经实现。

必测项：

1. 正负面积的输入形状、全零、全正、全负及纯正弦解析例子；
2. 循环平移后正负面积不变（同周期重采样）；
3. 五相共享尺度保持弱相的零强度，不使用 per-phase denominator；
4. 先从 Clean 的 **k0..4 重构**，不是从无截断的原始 Clean 计算训练目标；
5. 输出与目标共用同一 `s_cur`，梯度能从 \(\mathcal L_{\mathrm{pol}}\) 回到 B1 的收缩参数；
6. 原 B1 的 `forward`、训练/验证 RNG、三损失基准、权重文件和评估代码不被覆盖；
7. 无目标域文件进入 training、gradient 或 checkpoint selection；
8. 训练时验证梯度占比，避免 \(\lambda_{\mathrm{pol}}\) 的量纲过小而不起作用，或过大而破坏原 B1。

## 9. 最精简的验证设计

主对照仅需：

| 方法 | 唯一差异 | 指标 |
|---|---|---|
| **B1（现有）** | 基线完整模型 | 四档 Macro-F1、逐类 Recall |
| **B1 + polarity loss** | 仅新增结构监督，推理结构不变 | 相同四档 Macro-F1、逐类 Recall |

固定新 Source-Cycle V2、42/43/44、源训练/选模协议与 10+50 训练轮数。重点核实：

- OUS(A)/OLS(A) 的极性区分是否改善；
- OUS(A) 与 OUS(A)+OUS(B) 的 B 相证据是否改善；
- OLS(A) 与 OLS(A)+OP(C) 的 C 相弱电流特征是否保持；
- 5/0 dB 是否进一步接近 LP4，而不丢失 B1 的 -5/-10 dB 优势。

**不得提前声称它一定提高 seed43；若 seed43 属于分类头实测域偏置，单个辅助损失未必改变错误。**

## 10. 两个候选怎样组合？需要吗？

本方案与 [复谐波 CVAST 候选](B1_COMPLEX_AST_PROPOSAL.md) 不冲突：CVAST 修改 **推理时的收缩算子**，Polarity 修改 **训练时恢复损失**。但不要一开始做同时替换收缩并增加 loss 的大改版。

最少路径：B1 对照 → 验证 CVAST 或 Polarity 中更有依据的一项 → 如独立有效再考虑融合；若均无收益，保留 B1。

## 11. 参考

- 原 B1：[代码](../scripts/hust_b1_phase_harmonic.py)
- 新源域划分与实验：[服务器报告](../records/source_cycle_benchmark/server_4728e9b/SERVER_RESULTS.md)
- IADC-Net：[DOI](https://doi.org/10.1016/j.ymssp.2026.114419)；[原论文作者仓库](https://github.com/GTH25/IADC-Net)
- CVAST 候选：[B1_COMPLEX_AST_PROPOSAL.md](B1_COMPLEX_AST_PROPOSAL.md)

**结论：这是有针对性的故障电流结构监督候选，而不是必加模块。是否继续的前提是数据确实支持稳定、可迁移的极性差异，并且恢复器确实需要这项额外约束。**
