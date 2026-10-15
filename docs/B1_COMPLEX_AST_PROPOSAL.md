# Candidate 1 — B1-CVAST：噪声校准的复谐波自适应软阈值

> **状态：设计规格（Design Proposal），尚未实现、训练或取得任何新指标。**
> 
> 基线代码：[`scripts/hust_b1_phase_harmonic.py`](../scripts/hust_b1_phase_harmonic.py)，新源域划分：Source-Cycle V2。
> 
> 主要启发：Gao et al., *An Interpretable Adaptive Denoising Convolutional Network for Intelligent Fault Diagnosis of Rotating Machinery*, **MSSP**, 2026, DOI [10.1016/j.ymssp.2026.114419](https://doi.org/10.1016/j.ymssp.2026.114419)，ASTD（Adaptive Soft Threshold Denoising）。
>
> **本项目不是复现 IADC-Net；不迁移其 Laplace/Mexhat/Morlet 三路卷积、EER 权重或原 ASTD 偏移实现。**

## 1. 要解决的具体问题

HUST 的五相 IM 开路故障数据输入为五相同步电流，一个完整电气周期重采样到 `[B,5,4096]`。现有 B1 保留每相 k=0,1,2,3,4 的复数 FFT 谐波，通过估计噪声方差和多相特征生成连续增益，将观察系数缩小，再重构周期波形，输入 MD 波形及五相共享幅值的双视图 ResNet。

Source-Cycle V2、Seed **42/43/44** 下（实测 Target-Dev Macro-F1 均值，%）：

| 方法 | 5 dB | 0 dB | -5 dB | -10 dB |
|---|---:|---:|---:|---:|
| LP3+ResNet | 94.68 | 92.14 | 82.89 | 62.90 |
| LP4+ResNet | 97.59 | 92.82 | 78.05 | 49.54 |
| B1 | 94.36 | 93.54 | 89.25 | 70.98 |

已观察到 B1 的极低 SNR 优势，但 **5 dB 未达到 LP4 的精度**。本提案的明确假设是：对噪声主导的复谐波执行更有选择性的、具有显式零区间的收缩，可能改善噪声条件下的故障表征；但也可能误删微弱真实谐波、进一步损害中高 SNR。**目前没有证明现有 B1 的误差由“连续收缩不够果断”造成。**

## 2. 原论文哪些思想可以迁移、哪些不能

IADC-Net ASTD 将基于特征绝对值统计的动态阈值应用于小波特征系数。其论文版本还有非对称阈值偏移，但对本项目复谐波不宜原样照搬：

- **保留**：“阈值不固定，而是由当前特征和噪声状态决定”；允许明显不可靠的系数收缩到零。
- **改造**：输入是 **复谐波** 而非实数小波系数；阈值定义在复数 **模长** 上。
- **不照搬**：对实部和虚部独立执行非对称偏移，可能破坏复相位的循环移位等变性；原作者开源 ASTD 中的动态模块实例化和随机偏移还需另行质量核查，不直接复制。
- **避免空泛创新叙述**：经典复数软阈值并非新颖运算。论文价值必须来自其噪声标定、故障结构适配及经验证的实际收益。

## 3. 现有 B1 的输入及关键张量

设 `B` 为 batch，`p=1..5` 为五个电流相，`k=0..4` 为每周期复谐波阶次。现有 B1 的数据流：

| 中间量 | 维度 | 说明 |
|---|---|---|
| `raw` | `[B,5,4096]` 实数 | 含噪/干净单周期电流 |
| `full = rfft(raw, norm="ortho")` | `[B,5,2049]` 复数 | 全部可用频率信息，仅供输入侧统计 |
| `Y = full[:,:,:5]` | `[B,5,5]` 复数 | k0..k4 观察谐波 |
| `s` | `[B]` 正实数 | 五相 k1 **幅值中位数**，同一样本五相共享 |
| `Z=Y/s` | `[B,5,5]` 复数 | 无量纲归一化观察谐波 |
| `v_hat` | `[B,5,5]` 正实数 | B1 已有高频 k20..80 噪声代理 + 可学习逐阶校准，**是相对于 s 的无量纲功率** |
| `correction` | `[B,5,5]` 实数 | B1 的多相 Gram 相关与幅值特征 Conv1d 输出 |
| `S_hat` | `[B,5,5]` 复数 | 待恢复系数 |

**量纲提醒（重要）**：`v_hat` 在当前代码中由 `|full|²/s²` 构造。直接用 `sqrt(v_hat)` 与原始复谐波 `|Y|` 比较属于单位不匹配；必须在 `Z=Y/s` 空间收缩，或将阈值乘回 `s`。

## 4. 原 B1 与候选 CVAST 的唯一关键差异

### 4.1 原 B1 的收缩函数

当前 B1：

```text
signal_power  = |Y/s|^2
log_snr       = log(signal_power + eps) - log(v_hat + eps)
correction    = 2 * tanh(Conv1d(features))
gain          = sigmoid(log_snr + correction)
S_hat         = Y * gain
```

在忽略修正项时，属于近似 `|Z|²/(|Z|²+v_hat)` 的连续收缩形式；每个非零系数在数学上仍保留非零残量。

### 4.2 新候选：噪声校准的复数模长软阈值

对无量纲复系数 `Z[p,k]`：

\[
\tau_{p,k} = \alpha_{p,k}\sqrt{\widehat v_{p,k}},\quad
\widehat Z_{p,k}=\left(1-\frac{\tau_{p,k}}{|Z_{p,k}|+\varepsilon}\right)_+Z_{p,k},
\quad \widehat S_{p,k}=s\,\widehat Z_{p,k}.
\]

其中 \((u)_+=\max(0,u)\)。候选阈值系数使用 **B1 已有 adjustment head** 的输出；可以增加最多 5 个逐阶偏置参数 `b[k]`：

\[
\alpha_{p,k}
=\alpha_{\min}
+(\alpha_{\max}-\alpha_{\min})\,
\sigma(b_k+\Delta_{p,k}).
\]

初始候选值仅作为**预注册研究配置**：`alpha_min=0.05`、`alpha_max=2.0`、`alpha_init=0.25`；根据上述 Sigmoid 反函数初始化 `b[k]`，`adjustment` 的最后层继续零初始化。**不得按 Target-Dev 指标搜索 alpha 或阈值超参数**；任何进一步选型只能依据源域训练/验证。

采用已有的噪声估计 `v_hat`、多相 Gram 特征与卷积编码器，不另外叠加注意力或新 Transformer。

### 4.3 数学性质

1. **相位保持**：\(\widehat S_{p,k}=\gamma_{p,k}Y_{p,k}\)，其中 \(\gamma_{p,k}\in[0,1]\) 为实数；对非零保留项不旋转观察复数相位。
2. **零输入保护**：如果 \(Y_{p,k}=0\)，则 \(\widehat S_{p,k}=0\)，不会生成凭空出现的复相量，保护弱相的零输入性质。
3. **显式置零**：当 \(|Z_{p,k}|\le \tau_{p,k}\) 时置零，这是与原 B1 的核心差异。
4. **无放大**：\(|\widehat S_{p,k}|\le |Y_{p,k}|\)；但这并不等价于“恢复误差一定更小”。
5. **周期移位等变**：由于阈值网络输入为相间复 Gram 的共同相位不变量、能量和噪声代理，输入共同循环平移时同阶增益保持不变，复谱输出按原阶次相位旋转。需要以单元测试验证；后续分类 CNN **不自动具有**同样的性质。
6. **DC k0 特别注意**：k0 带有正负半周失衡/直流偏置信息，硬置零可能有害。V0 首先沿用“所有 k0..4 一致处理”的清晰对照，同时单独报告 k0 的置零率和类别影响。若 k0 确实成为主要失败因素，再单独定义后续版本；不要在首次对照中同时加入 DC 特例、阈值保护等多项修改。

## 5. 与原 B1 公平对比的固定项

**保持不变：**

- Source-Cycle V2：63 个仿真原始文件、13 组工况；Train=962 周期、Val=221 周期、间隔 78 周期；健康片段去重策略不变。
- 五相通道顺序、完整电气周期、FFT `norm="ortho"`、k0..4 截断。
- 噪声代理 k20..80、逐阶正功率校准、Gram 特征及 Conv1d 宽度/深度。
- iFFT → **MD 波形 + 五相共享标度波形** 两视图输入 → 原有 ResNet。
- 10 轮仅源域去噪器预训练 + 50 轮联合训练；Clean/Noisy CE、complex/phase-strength/noise-calibration 三项恢复损失权重与 B1 一致。
- 源域训练噪声 10/15/20/25/30/35/40 dB；源验证 Clean 和 0/5/10/20/30 dB；选模沿用源域 Macro-F1 优先 + 源域 CE 平局判定；seed **42/43/44**。
- 实测 Target-Dev 5、0、-5、-10 dB 仅在源域选模后推理；实测 Clean 仅作为分析条件。**无目标域训练或目标域选模。**

**唯一核心操作改动：** `sigmoid(log_snr+correction)` → 由同一特征网络输出阈值系数的复谐波模长 soft-threshold。少量 `b[k]` 参数的增加需在比较时报告。

## 6. 实施接口及质量要求

建议新建模型文件而非修改历史 B1：

- `scripts/hust_b1_complex_ast.py`：`B1ComplexASTNet`，与 `B1PhaseAwareRecoveryNet` 兼容的 `forward(raw,return_recovery=False)` 接口。
- 原 B1 完整代码及实验结果不动；在新研究分支中放独立 experiment runner、冻结配置与测试。
- 建议为阈值函数实现纯 PyTorch 复数向量化操作，不使用手动 `detach()` 切断梯度，不使用随机阈值偏移。
- `eps`、clip 等数值常量集中定义、由源域数据规模和浮点精度检查，不在实测结果上调。
- 必测：`[B,5,4096]`→`[B,6]`；复数张量与 real iFFT 的 dtype；k0 虚部为零；`Y=0` 不生成新的谐波；`0<=gain<=1`；小幅值系数可置零；梯度能传回阈值网络；同周期循环移位等变；有限值/极小能量相；CPU/GPU 确定性；source-only 数据隔离。
- 日志额外记录 `alpha`、`tau`、各相/阶增益、零化率、以及 k0 的置零率；统计不要仅凭平均 gain 判断模型可靠性。

## 7. 评价与明确的否决条件

**主指标：** 新划分、3 seeds（42/43/44），实测 5/0/-5/-10 dB Macro-F1、每类 Recall。比较 B1、LP3、LP4。不必增大网络或再跑六个种子。

**诊断指标（辅助）：**

- 对 Clean/已知残差派生目标样本，在 **离线评估** 中计算恢复后的前五阶复数误差；不用于训练或超参数筛选。
- 不同故障、不同相、不同 k 的 `gain/tau/zero_fraction`，尤其在 OUS(A) 及 OUS(A)+OUS(B) 中 A/B 相的可靠性。
- 5/0 dB 性能是否因过度零化明显下降，-5/-10 dB 的 OUS/OLS 与复合故障 Recall 是否改善。

**否决规则：**

1. 不能仅以更少的能量、更平滑的波形或更高的零化比例宣称抗噪有效。
2. 如果多数 seed 在 5/0 dB 明显退化而 -5/-10 无增益，终止该候选，不强行扩展。
3. 若软阈值造成较大比例弱故障谐波被误删，返回 B1，而不是叠加另一层阈值网络补救。
4. 如果方法有效，仍需承认复数软阈值本身是既有方法，不能将它包装为全新数学算子。

## 8. P1 现有 B1 恢复审计（推荐，但非运行本候选的硬依赖）

无需重新训练 B1：加载 42/43/44 的源域选中权重，实测条件逐档前向，提取已有 `gain`、`noise_variance`、`recovered`，对照已知 Clean 派生复谱，生成一张每相/每阶恢复误差与 gain 表。

目的是提前判断：究竟是“连续收缩太弱”还是“高 SNR 已存在过度收缩”、噪声代理是否离谱，以及什么谐波与 seed43 的错误类别有关。此审计是**解释性操作，不是要增加论文中的新基线**。

## 9. 文件与引用

- 当前 B1：[`scripts/hust_b1_phase_harmonic.py`](../scripts/hust_b1_phase_harmonic.py)
- 新划分已核验报告：[`records/source_cycle_benchmark/server_4728e9b/SERVER_RESULTS.md`](../records/source_cycle_benchmark/server_4728e9b/SERVER_RESULTS.md)
- 原 IADC-Net：[DOI](https://doi.org/10.1016/j.ymssp.2026.114419)；作者公开代码 [GTH25/IADC-Net](https://github.com/GTH25/IADC-Net)
- 第二独立候选：[`B1_POLARITY_PRESERVING_PROPOSAL.md`](B1_POLARITY_PRESERVING_PROPOSAL.md)

**结论：** 这是一个成本较低、变量相对单一的机制对比实验，不预先声称会涨点，也不把 ASTD 当作论文必需模块。
