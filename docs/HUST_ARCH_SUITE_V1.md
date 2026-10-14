# HUST 九模型结构优化对照 V1：一次性训练、严格单变量消融

## 研究目的

根据已完成实验，V5 的频域约束在 0/5 dB **合成极强噪声**下较稳定，但 ResNet1D 在 Clean/10 dB 表现更强。检验：

1. 保留 V5 完全相同的谐波前端，A′ **加宽**是否提高跨域特征表达能力？
2. 在加宽基础上 **加深**是否继续增益？
3. 用相同参数、相同随机初值的普通/残差双卷积模块，仅切换 **identity shortcut** 时有无收益？
4. 在第一层替换为 **多尺度并行卷积**有无独立收益？与残差是否有交互？
5. 直接把 LP3 / LP4 / NA-HSF **放在既有 ResNet1D 编码器之前**能否结合深层特征提取与低阶谐波抗噪？

**不以轻量化作为实验目标；先看 Clean/10/5/0 dB Macro-F1、逐类 Recall 与不同随机初始化表现。**

## 1. 结构矩阵（完整 9 臂）

已有 V5（25,661参数）和 ResNet1D（241,702参数）只作为历史对照，**不重训**。本轮训练 9 类模型 × seeds [42,43,44] = **27次独立训练**。

| 模型名 | 前端 | CNN 主干 | 主要变量 | 参数量（本地 CPU Smoke） |
|---|---|---|---|---:|
| `v5_w` | 原 V5 NA-HSF | A′ 3 层加宽 32/64/64 | 相对 V5 只改宽度 | 63,213 |
| `v5_wd_plain` | 同上 | 加宽 + 相同64通道双卷积 | 深度效应 | 88,045 |
| `v5_wd_res` | 同上 | 相同双卷积 + identity shortcut | 与上一项严格等参数 | 88,045 |
| `v5_w_ms` | 同上 | 加宽 + 多尺度 Stem | 对加宽结构的多尺度独立效应 | 63,693 |
| `v5_wd_plain_ms` | 同上 | 多尺度 + 双卷积无 shortcut | 多尺度下的无残差对照 | 88,525 |
| `v5_wd_res_ms` | 同上 | 多尺度 + 双卷积 + identity shortcut | 多尺度×残差 | 88,525 |
| `resnet_lp3` | LP3 (k0..3=1，其余0) | 原 ResNet1D，32/64/128 | 固定强滤波 + ResNet | 241,702 |
| `resnet_lp4` | LP4 (k0..4=1，其余0) | 原 ResNet1D，32/64/128 | 固定较宽滤波 + ResNet | 241,702 |
| `resnet_nahsf` | 原 V5 NA-HSF (第4阶 α(x)) | 原 ResNet1D，32/64/128 | Gate/ResNet 结合 | 241,743 |

### 公平控制

- **B2 vs B3，B5 vs B6**：双卷积内 Conv、BN、激活完全相同，差异只有 `x+F(x)` 对比 `F(x)`；同seed的可学习权重**逐张量相同**，参数量完全一致。不同前向和训练梯度产生的效果差可解释为 shortcut 干预的效应（在本固定实验协议与训练随机流条件下）。
- **B1 vs B4**：两种 Stem 的输出通道数均32、Conv/pool后输出长度均56；多尺度 Stem 改变卷积核及参数量，因此这是多尺度架构效应，**不是等参数对照**。
- **B3 vs B6**：相同深层编码器；只替换第1层 Stem 结构，以评估多尺度贡献；存在 Stem 参数量差，需在论文中披露。
- **ResNet 对照**：三种新 ResNet 使用相同完整原始 ResNet1D 模块、相同随机种子初始化，**只改变频域前端**；不会把128点波形再执行一遍 MD。相对于已有 ResNet1D（无滤波），新增对照测频域约束的贡献。
- 所有模型仅使用正常的 Clean/Noisy CE；不使用目标域标签、目标伪标签、域对抗、Decoder/DML或任何信息泄漏。

### 多尺度位置对齐的关键细节

单尺度 A′ 16点无填充卷积，128点输入后长度 **113**；后接 MaxPool(2) 变 **56**。多尺度 Stem 采用奇数卷积核 `7/15/31`、各自 `padding=k//2` 得到128点输出，然后在**相同时间位置**裁剪 `[7:120]`，三个分支输出统一长度113；拼接通道数 `8+12+12=32`；后接相同 BN、LeakyReLU、MaxPool(2)。其他两层 kernel4 valid+pool 后为64×11，因此加深前的 Flatten 特征是704维。多尺度分支都对准同样的中心时间坐标；其采样中心和原偶数k16有效卷积差半个点，不声称卷积核几何完全一致。

### 谐波前端不改变

原 V5 依赖 `v5_nahsf.NoiseAwareHarmonicFusion`（3→8→1，输入频谱描述符来自当前样本）。第0–3阶幅度增益1，第4阶`alpha(x)`，第5阶及以上0；保留复频谱相位。LP3复用 `v2_paifilter.HarmonicShaping('lowpass3')`，LP4使用等价确定性实增益 mask。FFT `norm='ortho'`，IFFT一致。均在原始**4096点周期**上完成，之后仅执行**一次**现有 `MDNormalization`（中心化、低电流相分支、线性重采样128点）。ResNet 用已有 `.stem / .layers / .pool / .classifier` 对已归一化128点输入推理，不调用它要求4096点原始输入的 `forward`，避免重复归一化。

## 2. 冻结数据及训练/评估协议

| 项目 | 冻结设置 |
|---|---|
| Source-Train | 34条 simulation recording，738个五相完整电周期 |
| Source-Val | 15条 simulation recording，302个周期，记录与训练无交叉 |
| Source noise train | 每轮重新生成 10/15/20/25/30/35/40 dB，整记录、逐相 RMS、同一 seed/epoch 共用噪声随机流 |
| Loss | `0.5 CE(Clean) + 0.5 CE(Noisy)`；Clean+Noisy拼接后同一次前向（保持 BN 行为） |
| Optimization | seed42/43/44、50epochs、batch32、Adam lr1e-3、wd0、每轮24步 |
| Source-Val | Clean＋固定0/5/10/20/30 dB（0/5从已有 `generated/source` MAT） |
| Checkpoint | 所有50轮中，源Clean F1距最高分≤1pp；从合格epoch选五档Source-Val噪声平均Macro-F1最高者；并列以Clean高及epoch早打破 |
| Target-Dev | 88条记录、每条件2474周期；Clean、作者原实测 10–40dB 七档，及合成0/5 dB |
| Forbidden | 使用实测Dev标签训练、调超参数或选epoch；读取实测347条Final-Holdout或源保留Test |

**重要说明**：Target-Dev 的 0/5dB 是从作者 10dB 实测残差按幅度放大而来，**不是独立采集**；应单独标记为极端噪声压力实验。当前已进行多轮Dev研究，最后结构选定之后仍需隔离 Final-Holdout 做一次独立验证。即使只在文章主表列3个seed，已有的另外3个V5初始种子结果也应在研究记录中保留，并谨慎表述稳定性。

## 3. 文件位置与一次性执行

新文件：

- `scripts/hust_arch_suite_models.py`：9模型统一工厂、MS Stem、恒定LP4、参数完全一致的残差对照。
- `scripts/hust_arch_suite_experiment.py`：同一套源域train/val、Target-Dev评估、逐种子与逐类报告。
- `configs/hust_arch_suite_v1.json`：冻结矩阵与训练超参数，入口会强验证。
- `tests/test_hust_arch_suite.py`：结构/输出/参数/随机初始化/FFT/梯度/防覆盖/源域选模检查。
- `scripts/run_hust_arch_suite_server.sh`：运行一次完成全部9模型、3种子。

在 **hust-SNR repo 根目录**，使用此前已验证的 CUDA/Python 环境：

```bash
# 新分支切换
 git fetch origin
 git switch research/architecture-sweep-v1
# 指向既有真实数据根目录，不要照抄路径占位符
 export SOURCE_STRONG_ROOT="/path/to/existing/strong_data_root"
 export TARGET_SNR_ROOT="/path/to/original_target_SNR_parent"
 export TARGET_STRONG_ROOT="/path/to/existing/strong_data_root"
 # 可选，默认 python
 # export PYTHON_BIN="/path/to/venv/bin/python"
 bash scripts/run_hust_arch_suite_server.sh
```

脚本依次：25项pytest、模型梯度smoke、**27组源域训练**、**270组Target-Dev推理**、完整结果统计。运行时输出为 `outputs/hust_arch_suite_v1/`，不接触 V5/ResNet 的旧权重目录；任何已存在的训练目录或输出都会拒绝覆盖。允许单独通过CLI指定 `--methods <subset> --seeds <subset>` 分批/断点继续；**不要**对已经训练的条目重复运行。全体训练完成后统一运行 evaluate + summarize。

输出结构：

```text
outputs/hust_arch_suite_v1/
├── models/<model>/seed<seed>/{best.pt,history.json,run.json}
├── evaluations/<model>/seed<seed>/target_dev_<condition>.json
└── reports/
    ├── PER_SEED_RESULTS.csv
    ├── PER_CLASS_PER_SEED.csv
    ├── SUMMARY.csv
    └── SUMMARY.json
```

新 checkpoint 包含正确方法名、seed、实验协议和仅源域选模的记录；不覆盖旧文件。代码从未创建 Final-Holdout 的加载入口。

## 4. 如何分析结果与决定后续论文方法

优先比较：

- `v5_w` vs 历史V5：加宽，注意 Seed 同名不代表参数完全相同，且源域选模可能发生改变；这是结构替换对照。
- `v5_wd_plain` vs `v5_w`：加深；`v5_wd_res` vs `v5_wd_plain`：**残差唯一开关**（同参数及可学习初始权重）。
- `v5_w_ms` vs `v5_w`：独立多尺度；`v5_wd_res_ms` vs `v5_wd_plain_ms`：多尺度背景下的残差效应；`v5_wd_res_ms` vs `v5_wd_res`：多尺度与残差组合效果。
- `resnet_lp3` vs `resnet_lp4` vs `resnet_nahsf` vs 历史未滤波ResNet：是否有谐波先验收益，以及Gate是否胜于简单固定低通。

不应只选择一个Seed的最好成绩；优先看Clean/10/5/0 dB Macro-F1均值、样本SD、逐类Recall、故障判Healthy数量和逐seed配对增益。**即使新模型优于旧V5，也不自动证明噪声自适应门控有独立贡献**：最终骨干确定后还需相同骨干 LP3/LP4/固定alpha 的比较；ResNet+LP3/LP4/NAHSF三臂已经覆盖部分机制检验。

## 5. 实际测试状态

本地 CPU 依赖接口替身上完成模型结构、张量尺寸、FFT、MS对齐、权重一致性、前后向、参数量及协议测试：**25 passed**；九模型 CLI smoke 全部通过。语法检查和shell `bash -n` 通过。这些测试替身仅用于在缺少原 MAT/源仓库依赖的本地容器检查新代码；**真实现有代码/真实数据的GPU集成检查和正式训练必须在服务器运行**，不应将 Smoke Test 称作实验已经完成。
