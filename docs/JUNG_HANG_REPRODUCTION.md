# Jung Modified CNN 与 Hang WCNN 复现说明

## 1. 目标与复现边界

本目录实现两个论文启发式对比模型，适配当前 HUST 五相 IM-AA 六分类、仿真域训练到实验域开发集评估的流程：

- Jung：扩张 Conv1D、残差块、GAP，以及训练期高斯白噪声增强。
- Hang：五相电流 VSD 轨迹图、固定 Haar DWT 的 LL 子带池化、五层二维 CNN。

两篇论文均未公开足以逐层重建的完整代码与全部超参数。因此代码属于**尽可能忠实于论文核心机制的近似复现/任务适配**，不是声称与作者实现逐参数一致。

## 2. 文件

- `scripts/jung_hang_models.py`：两个模型、固定 DWT 池化、五相 VSD 与轨迹图栅格化。
- `scripts/compare_jung_hang.py`：训练、目标 Dev 评估、结果汇总和 synthetic smoke test。
- `tests/test_jung_hang_models.py`：CPU 单元测试。
- 运行输出位于 `outputs/jung_hang_comparison/`，不提交模型权重或预测文件。

## 3. 实现细节与明确的适配选择

### Jung Modified CNN

网络输入为现有预处理输出 `[B,5,128]`。网络包含：

1. 5→32 通道、kernel=3、dilation=3 的一维卷积；
2. BatchNorm + ReLU；
3. MaxPool1d(2)；
4. 两个 3-tap Conv1D 构成的残差块；
5. Global Average Pooling；
6. 六类线性分类头。

该重建版本有 7,110 个可训练参数。论文 Table III 的 7,213 参数是作者原始输入与结构下的结果，不能要求五相六类适配模型保持同样参数量。

原文用 THD+N 定标白噪声。当前仓库的跨域协议按 SNR 记录，因此这里采用**按每条仿真训练记录、每个相电流通道定标的 SNR-AWGN**，每个训练 epoch 从 10/15/20/25/30/35/40 dB 中抽取一档，先在完整原始记录上加噪，再按同一周期边界切窗并做原预处理。每个 batch 对 Clean 与 Noisy 两个视图各计算交叉熵，损失取二者平均。它是针对现有协议的增强适配，不应在论文中称为精确复现作者的 THD+N 设定。

### Hang WCNN

模型输入为 `[B,2,64,64]`，两个图像通道分别对应五相 VSD 的 `alpha1-beta1` 与 `alpha3-beta3` 子空间轨迹。处理步骤：

1. 对每个周期窗口的五相原始电流逐相减去窗口均值以移除 DC 偏置；
2. 使用标准五相 VSD 投影，生成基波与第三谐波/第二子空间轨迹；
3. 四个 VSD 分量共用一个幅值尺度，不单独放大弱谐波平面；
4. 轨迹以固定坐标范围栅格化为两个 64×64 图像通道；
5. 本实现选择五层 Conv2D 通道数 4/8/16/32/64（原文未披露完整逐层参数表），在前两个卷积阶段后使用固定 Haar LL 池化；
6. 展平并使用六类线性分类头。

DWT 池化使用固定的 2×2 全 1 LL 核、步长 2，不含可训练参数；其余高频子带被丢弃。该实现保持原文的“轨迹图 + 固定小波池化”机制，但将三相 Clarke 变换替换为五相 VSD，并使用两个子空间作为双通道输入。

**VSD 角度假设：** 默认相轴为 0°、72°、144°、216°、288°。这假设五相绕组均匀分布；如果实际数据的相序或相轴定义不同，必须先核对电机资料，再修改 `five_phase_vsd` 的 `phase_angles_deg`，不能仅凭分类结果反推变换正确。

### 公平的训练与 checkpoint 选择

两个模型共用以下规则：

- 仅使用冻结的仿真 Source-Train / Source-Val 文件划分训练和选择 checkpoint；
- Clean 验证 Macro-F1 距离训练期间最高 Clean F1 不超过 1 个百分点的 epoch 才有资格入选；
- 在合格 epoch 中选择 Source-Val 10/20/30 dB 平均 Macro-F1 最高者；平局时依次选择 Clean F1 更高、epoch 更早者；
- Jung 使用 Clean 与 Noisy 训练视图；Hang 使用 Clean 训练视图。验证噪声仅用于统一的 source-only checkpoint 选择，不用目标域标签调参；
- 目标评估只读冻结的 88 条 Dev 记录，使用相同的 Clean 周期边界评估 Clean 与七档 SNR；
- 不读取、不生成、不评估 347 条 Final-Holdout 的数据；
- 复用现有预处理和六类标签，报告 Macro-F1、Accuracy、Balanced Accuracy、逐类 F1/Recall、混淆矩阵、正常类误报率与故障漏判为正常的数量。

## 4. 运行方式

先确保仓库规定的原始数据位于服务器本地，并已生成 `outputs/gate0_v1/source_train.npz`、`source_val.npz` 及相应窗口元数据。原始 MAT、RAR、模型权重不应提交到 GitHub。

从仓库根目录执行：

```bash
python scripts/compare_jung_hang.py smoke
pytest -q tests/test_jung_hang_models.py
```

先做一次短训练检查：

```bash
python scripts/compare_jung_hang.py train --methods jung hang --seeds 42 --epochs 2
```

正式训练（默认 50 epochs、batch size 32、三个种子）：

```bash
python scripts/compare_jung_hang.py train --methods jung hang --seeds 42 43 44
```

评估目标 Dev。若 SNR 文件存放在外部目录，`--snr-root` 指向包含 `SNR_10/`、`SNR_15/`、…、`SNR_40/` 的父目录：

```bash
python scripts/compare_jung_hang.py evaluate --methods jung hang --seeds 42 43 44 --snr-root "/PATH/TO/Five-phase-IM-AA-experimental-domain-different-SNR-data-set"
python scripts/compare_jung_hang.py summarize --methods jung hang --seeds 42 43 44
```

若 SNR 数据已按 `data/target_dev/snr10/` 等结构放在仓库本地，则可省略 `--snr-root`。缺失配对 SNR 文件时程序会报错，不会用合成目标噪声代替官方评估数据。

训练或评估会拒绝覆盖已存在的同名结果目录/文件。需要重跑时使用新的 `--output` 路径，例如 `--output outputs/jung_hang_comparison_rerun1`。

## 5. 输出文件

每个方法/种子会产生：

- `models/<method>/seed<seed>/best.pt`：source-only 规则选择的 checkpoint；
- `history.json`：逐 epoch 训练损失、Clean 验证 F1 与各噪声档位验证 F1；
- `run.json`：种子、训练轮数、参数量、选模规则和数据使用声明；
- `evaluations/<method>/seed<seed>/target_dev_<condition>.json`：逐窗口真值、预测、混淆矩阵与指标；
- `SUMMARY.json`、`PER_SEED_RESULTS.csv`：按种子和 SNR 汇总的均值与标准差。

## 6. 参考论文

1. J. Jung et al., “Robust Open-Switch Fault Diagnosis of Three-Level NPC Inverters Based on Data Augmentation With White Noise Injection,” *IEEE Transactions on Power Electronics*, 40(2), 2025. DOI: 10.1109/TPEL.2024.3488094.
2. J. Hang et al., “Robust Open-Circuit Fault Diagnosis for PMSM Drives Using Wavelet Convolutional Neural Network With Small Samples of Normalized Current Vector Trajectory Graph,” *IEEE Transactions on Industrial Electronics*, 70(8), 2023. DOI: 10.1109/TIE.2022.3231304.
