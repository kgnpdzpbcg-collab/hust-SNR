# HUST-SNR V4 / HPASF V0：四组公平对照，先实验后决定论文方向

**运行状态（2026-10-12）**：实际提交`b4801df`完成四组×三个种子×50轮训练及96组实测评估；CUDA中位数算子替换为等价的稳定排序取中值，严格确定性保持开启。33项测试、48组源Val分数及96组实测预测复核通过。[服务器完整记录](../records/v4_hpasf/server_b4801df/SERVER_RESULTS.md)。

## 1. 研究假设及其证据边界

V2 Target-Dev 上固定三阶低通＋A′ 达到 10 dB Macro-F1 **95.7868%**（3 seeds 平均）；优于 V2 恒等输入 **90.2685%** 和近恒等初始化频域门控 **91.2703%**。**这些是旧 V2 的结果，不是 V4 成绩。**

前期 HUST 五相真实数据探索发现，绝大多数 Clean 电流非直流频谱能量集中于 1–3 阶；第4阶在部分复合开路类别仍能观察到高于噪声底的能量；12–30 阶的中位功率可作为候选噪声底代理，在高 SNR 可能高估噪声。**上述只是模块设计依据，不能推出第4阶能提供额外独立判别信息。**

同时，V2 低通模型的 10 dB 部分错误在 Clean 中已经存在，因此“更好降噪”未必解决其主要误差。V4 要检验是否可以在保留已有效的三阶低通基础上，有条件地恢复第4阶并取得额外分类收益。

## 2. 完全冻结的四组（A′＋C，不包含 CAE Decoder）

所有模型均为原 HUST CAE-Classification 的 Encoder+Classifier 简化版 A′，即 V2 的 **25,620 参数 1D-CNN**；无 Decoder、重构 MSE、度量学习或 V1 一致性损失。核心数据路径：`原始五相电流完整电气周期 → 4096均匀插值 → FFT/频谱选择 → IFFT → V2 MD归一化 → 五相×128点 → A′ → 六类`。滤波前不使用 Clean 参考电流。

| CLI `--methods` | k=0–3 增益 | k=4 增益 | k≥5 增益 | 谱参数 | 目的 |
|---|---|---|---|---:|---|
| `lowpass3` | 1 | 0 | 0 | 0 | 复验 V2 强基线，**直接调用 V2 原低通模块** |
| `lowpass4` | 1 | 1 | 0 | 0 | 第4阶直接放开是否有益 |
| `static4` | 1 | `sigmoid(a)` | 0 | 1 | 仅学习**跨样本常量**是否有益 |
| **`hpasf`** | 1 | `sigmoid(a) × r4(x)` | 0 | **1** | 输入自适应频谱可靠性是否额外有益 |

第4阶增益所有五相共享，为实数且不大于1，不旋转第4阶的跨相复数相位；k≥5 不参与重构。`static4/hpasf` 的 `sigmoid(a)` 初始化为 **0.05**，使两种可学习模型均从接近 lowpass3 的前置滤波开始，保留可训练梯度。对 `hpasf`，因为乘有 `r4`，实际初始增益不超过 0.05。该门控不是 FilterNet PaiFilter 原样复现；借鉴的只是频域乘性选择思想。总参数：前两组 25,620；后两组 25,621。

## 3. HPASF 唯一新增公式（事前固定）

`X = rFFT(x, norm='ortho')`；每个样本在五相上合并同阶功率：

\[
P_k(x)=\frac15\sum_{p=1}^{5}|X_{p,k}|^2,\qquad
\widehat N(x)=\mathrm{median}_{k=12,\ldots,30}P_k(x).
\]

\[
r_4(x)=\mathrm{clip}\left(\frac{P_4(x)-\widehat N(x)}{\max(P_4(x),\epsilon)},0,1\right),\quad
g_4(x)=\sigma(a)r_4(x).
\]

中位数仅来自**当前输入样本**，不使用真实/预测故障标签、对应 Clean 信号、已知目标 SNR 或 Dev 统计。五相功率平均与 `12..30` 阶均在实验前固定，不能按 Target-Dev 得分反复调整。`r4` 在反传中 `detach/no_grad`，频率标量 `a` 依然通过 CE 正常更新；这是为了隔离基于样本计算的不可导中位数部分，并不声称其为可学习噪声估计器。高阶信号可能污染噪声底，**此模块为待验证假设**。

只保留一个可学习标量（不引入光谱平滑正则、额外频段、频率注意力或其他可调阈值），从而隔离 `static4` → `hpasf` 的增益是否来自输入可靠性估计。

## 4. 对齐 V2 的源域训练与目标域评估

- 源 Train = 34 条记录 / 738 周期；源 Val = 15 条记录 / 302 周期；88 条 Target-Dev（2474 固定窗口）仅用于评估；347 条 Final-Holdout 禁止触碰。
- 原 V2 `raw_cycles` 数据插值与完整记录相 RMS 高斯增强函数完全复用；每条记录每轮从 SNR 10/15/20/25/30/35/40 dB 抽取，三模式同 seed/epoch 同一增强 RNG 和相同 batch 顺序。
- 同种子 42/43/44、初始 CNN 参数字节相同、Adam lr=0.001、weight_decay=0、batch32、50 epochs；损失为 `0.5*CE(clean)+0.5*CE(noisy)`，**不附加任何 HPASF 正则项**。
- **仅根据源域** Val Clean 和固定模拟 10/20/30 dB 选 checkpoint：Clean-F1 位于整个训练过程最优值的 1 pp 内，选源域 noisy-F1 均值最高的 epoch；并列选 Clean 更高且 epoch 更早；与 V2 完全一致。
- 原 MAT 为正式 Target-Dev 测评输入，备选 `--dev-export-root` 会读 float32 导出 NPZ，可能与 MAT 存在临界舍入差异。评估 Clean + 七档 SNR，并输出六类 F1/Recall、混淆矩阵、健康误报、故障判 Healthy、逐周期预测，汇总三个种子的配对差值。
- V4 重新训练 `lowpass3` 是同一管线控制组（期望与 V2 同模式表现相近，需通过实际数值复核），不能只拿 V2 历史 95.79% 和 V4 一轮结果比。
- 由于 HPASF 假设研发阶段看过 Target-Dev 的信号与诊断结果，**只可称训练权重与选模不使用目标域**，不能把完整方法研发标榜为对目标域完全盲检。

## 5. 服务器运行（仓库根目录）

```bash
# 新研究分支，不修改 V1/V2/V3
 git fetch origin
 git switch research/hpasf-v0
 python scripts/v4_hpasf.py
 python scripts/v4_experiment.py smoke
 python -m pytest -q tests/test_v4_hpasf.py tests/test_v4_experiment.py

 # 完整：四组×三个seed×50 epoch；禁止重跑覆盖同目录
 python scripts/v4_experiment.py train
 python scripts/v4_experiment.py evaluate --snr-root "/path/to/SNR_parent"
 python scripts/v4_experiment.py summarize
```

若只想确认 CUDA 可学习谱参数与确定性反传，先运行测试，再跑完整训练；`outputs/v4_hpasf/` 为唯一输出目录，不修改原 V2。若需要使用 Dev 导出 NPZ，改成 `evaluate --dev-export-root /path/to/dev_export`。

## 6. 基于结果决定下一步，避免反复试错

优先比较 10/15/20 dB 三种子 Macro-F1 **配对差值**、Clean 变化、每类 Recall（重点 OUS(A)、OLS(A)）、missed_fault_as_normal，以及低通3与低通4的区别。

- 若 `lowpass4` > `lowpass3` 且 `hpasf` > `static4`：说明第4阶信息有价值，自适应有额外可能贡献，继续研究可靠性与稳定性。
- 若 `lowpass4` > `lowpass3` 但 `hpasf` 无增益：优先采用经典低通策略或重新评估噪声底代理，无须堆网络。
- 若 `lowpass3` 仍最佳：承认额外第4阶无可靠收益，不得用“先验自适应”包装负结果。
- 任一模型如果提高平均值但显著增加故障→正常误判、牺牲 Clean 或仅一个种子有效，都不算稳健改进。

这些判据是方向性的，不将 1 个 Dev 数据分组的微小差异认作显著性结论；Final-Holdout 在确定最终设计后只运行一次。
