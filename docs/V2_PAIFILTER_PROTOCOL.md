# HUST-SNR V2: 轻量 PaiFilter 谐波整形 + 噪声增强 CNN

**状态（2026-10-12）**：已完成可运行的 V2 模型、源域训练脚本、Dev 测评入口和单元测试。**未**执行 3 seeds × 3 methods × 50 epochs 的正式 GPU 实验。Final-Holdout（347 条原始记录）保持隔离。

## 1. 来源：准确界定与原论文的关系

- 参考：Yi et al., *FilterNet: Harnessing Frequency Filters for Time Series Forecasting*, **NeurIPS 2024**。官方仓库：https://github.com/aikunyi/FilterNet ，原文件 `models/PaiFilter.py`。
- 官方 PaiFilter 在预测任务中使用共享可训练**时域循环卷积核** `w`，计算 `rfft(x) * rfft(w)` 后进行 `irfft`，并搭配 RevIN 和预测输出层。
- 本 V2 **不是官方 PaiFilter 原样复现**：改成五相共享、**实数非负不放大的频率增益**；删除 RevIN、预测头；置于 **原始电流归一化之前**，后接原 V1 诊断 CNN。参考思想是频域乘性滤波，不声称此种实数门控或任务迁移是 FilterNet 作者的原设计或天然保证抗噪。

## 2. 结构已经冻结

`五相原始电流 -> Clean-derived 完整周期 -> 4096 点均匀插值 -> FFT -> 实数频率门控 -> IFFT -> HUST 原论文的 MD 式极值中心化/低电流保护 -> 128 点线性插值 -> 现有 1D-CNN -> 六类输出`

| 组件 | 结构 | 可训练参数 |
|---|---|---:|
| 原 V1 1D-CNN | 卷积 5→16→32→32，FC352→50→6 | 25,620 |
| `identity` | 不做谱滤波 | 0 |
| `lowpass3` | 硬保留 DC 及 1–3 次谐波 | 0 |
| **`paifilter`** | 共享频率衰减掩码，前 32 阶独立增益，后面 96 个高频控制点线性插值；DC 恒为 1 | **128** |
| **V2 总计** | `paifilter`＋原 CNN | **25,748** |

`paifilter` 定义（与原论文的循环卷积核不同）：

\[
X_k=\mathrm{rFFT}(x)_k,\qquad
\widetilde{X}_k=g_kX_k,\qquad
\widetilde{x}=\mathrm{irFFT}(\widetilde{X}),\quad
g_0=1,\quad g_k=\exp[-\max(0,a_k)]\in(0,1].
\]

`a_k` 为 128 个可训练标量经过低阶独立／高阶插值展开的结果；初始均为 0.001，所以初始频率增益约 0.999。五相共享所有频率增益，**不对复数频谱作额外相位旋转**。输出到归一化之前；PyTorch 的 FFT/IFFT、`max/min`、插值连接到 CE 反传，实际 gradient tests 已核验。

**不包含**：V1 的特征余弦一致性损失、类标签条件滤波、目标域自适应、目标 SNR 估计、频段对目标域的搜索。

## 3. 固定消融与公平协议

三组 **同时使用相同**：Gate 0 冻结的 34 条 Source-Train（738 周期），15 条 Source-Val（302 周期），三随机种子 42/43/44、Adam 1e-3、50 epochs、batch32；每个源域记录按相使用全记录 RMS 生成 10/15/20/25/30/35/40 dB Gaussian 噪声，源域干净/加噪各 CE 权重 0.5（即 V1 M1），所有方法在相同种子/epoch 接收同一噪声和 minibatch 顺序。

- `identity` = **同一 4096 点新输入管线下**，无频域模块的重训 M1 基线（重要：它不是 100% 字节等同于之前用原分辨率极值的 V1 M1）。
- `lowpass3` = 4096 点上先硬选前三阶，然后归一化、CNN。
- `paifilter` = 可学习实数频率门控，然后归一化、CNN。

全部模型使用**仅源域**的 checkpoint 规则：每 epoch 在 Clean Source-Val，以及固定 10/20/30 dB 噪声 Source-Val 上测 Macro-F1；Clean 在最佳 Clean-F1 下 1pp 以内的候选中，选择带噪均值最高的 epoch；平分时比较 Clean F1，再选择更早 epoch。绝不使用目标 Dev 选择权重／epoch。`V1` 的历史结果只作参考，V2 内主要公平对照应使用三种方法全部重训的指标。

## 4. 输入预处理与 V1 一致性注意事项

V1 首先在**原始每周期时间点**计算每相 `(max+min)/2`，当中心化平均绝对值 <0.2 时除以100，否则除以最大绝对值，再插值到128点。V2 因为需要 FFT 处理且需要端到端微分，先将**原始物理电流**周期统一到4096点，再在滤波后以 PyTorch 实施同样的 MD 运算和插值。

实际上传的 88 条 Dev × 2474 周期对照（`identity` 与原 V1 MD 输出的整体输入 RMSE，**不是诊断F1**）：

- Clean：**0.00011955**；10 dB：**0.00421283**。
- 差异来自原生测量时间点与均匀4096点插值后的峰值估计略有变化，尤其含噪尖峰。**不得将 V1 已存 89.96% 当作 V2 identity 的实际性能**，更不能由波形相似推断分类一致。
- 低电流阈值及“除以100”分支不修改；输入各相极性和相间相位关系必须保留。

## 5. 服务器执行

先拉取 `research/paifilter-v2` 分支（本文件将被提交到该分支），并保留 `data/source/*.mat`、`data/target_dev/clean/*.mat`、`outputs/gate0_v1/*.json/.npz`。

```bash
python scripts/v2_paifilter.py
python scripts/v2_experiment.py smoke
python -m pytest -q tests/test_v2_paifilter.py tests/test_v2_experiment.py
python scripts/v2_experiment.py train --methods identity lowpass3 paifilter --seeds 42 43 44
python scripts/v2_experiment.py evaluate --snr-root "/PATH/TO/parent_of_SNR_10"
python scripts/v2_experiment.py summarize
```

正式训练与测评输出至 gitignored 的 `outputs/v2_paifilter/`，避免覆盖原 V1 模型结果。评估只读取 `outputs/gate0_v1/target_dev_clean_windows.json` 固定的88条 Dev 及相同周期边界；输出 Clean、10–40 dB 的逐周期真值、预测、混淆矩阵、Macro-F1、Accuracy、Healthy 误报。

也可在已经有用户导出数据的机器上评估（该入口只读取 88 条 Dev）：

```bash
python scripts/v2_experiment.py evaluate --dev-export-root "/path/with/dev_reference.npz_and_dev_snr*.npz"
```

注意：导出 NPZ 的电流是 float32，而服务器 MAT 原数据是 float64。两个评估入口可能在极个别临界样本有微小数值差异，正式结果以原 MAT 入口为准。

## 6. 完成验收及论文判断门槛

已完成功能性验证：前向形状、25,748 参数、频域相位保持、低频通过／高频抑制、实数门控不放大、低电流相不被无故放大、FFT→MD→CNN 的梯度通过、频率参数真的更新、原始电流插值与 source-only 噪声复现、拒绝覆盖旧运行目录、真实 Dev 波形兼容性检查。正式训练结果**未完成，尚不得宣称 V2 提升抗噪准确率**。

关注 10/15/20dB 三种子配对增益、Clean 保真、OUS(A) 与 OLS(A) 召回和健康误报；若仅与未重训的历史 V1 M1 比较，则不能证明门控模块收益。如果固定 lowpass3 已优于或不弱于 paifilter，说明更复杂的可学习增益不一定必要。
