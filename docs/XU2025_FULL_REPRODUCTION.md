# Xu et al. (TPEL 2025)：完整 CAE + DML 复现与公平扩展实验 V1

**状态：已实现论文结构与训练/评估入口；仅本地 CPU 正向、反向及单元测试已验证。尚未运行 HUST 正式多种子 GPU 训练，不存在新模型准确率结果。**

论文：Xu et al., *Fully Simulated Data-Driven Domain Generalized Method for Multiphase Converters Fault Diagnosis*, IEEE Transactions on Power Electronics, 40(1), 2099–2113 (2025). DOI `10.1109/TPEL.2024.3461159`。复现以用户提供的 15 页 PDF 为第一手依据。作者公开仓库 `XHX-HUST/Multiphase-converter-switching-device-open-circuit-fault-data-set` 只公开数据；本实现**不是作者的官方代码**。

## 1. 原论文到底有什么：与旧 A′ 的区别

| 方法组件 | 论文来源 | 本实现 | 旧 A′ |
|---|---|---|---|
| 一电周期窗及依转速适应采样 | 式 (5),(8) | 项目冻结的周期边界，见第3节 | 有 |
| 每相去偏 `(max+min)/2`、阈值 0.2 归一化 | 式 (6),(7)，Fig. 3 | `MDNormalization` / `prepare.preprocess` | 有 |
| Conv1d+BN+LeakyReLU×3, MaxPool×3 | Table III, Fig. 4 | `XuCAE.encoder` | 有 |
| **上采样×3、Conv1d+BN×4 的解码器** | Table III, Fig. 4 | **`XuCAE.decoder`** | **没有** |
| **电流重构损失** | 式 (11) | **`xu_joint_loss`** | **没有** |
| 352→50→6 分类器 | Table III, Fig. 4, 式 (12)(13) | `XuCAE.classifier` | 有 |
| **类内聚集-类间分离损失** | 式 (14)–(18) | **`xu_metric_losses`** | **没有** |
| 三目标联合训练 | 式 (19)(20)，Algorithm 1 | **分类+重构+DML** | **只有分类** |

模型主输入 `x∈R^[B,5,128]`，encoder 形状链：

`[B,5,128] → [B,16,113] → [B,16,56] → [B,32,53] → [B,32,26] → [B,32,23] → [B,32,11] → flatten [B,352]`。

Decoder 对应 Table III：`[B,32,11] → upsample [B,32,22] → conv [B,32,26] → upsample [B,32,52] → conv [B,16,56] → upsample [B,16,112] → conv [B,16,128] → conv [B,5,128]`。分类器 `352→50→6`。复现网络总参数以实际运行 `parameter_counts()` 为准：**35,475 个可训练参数**。若加入 NA-HSF 前置门控，再增加 41 个参数，合计 **35,516**。此处的参数是模块参数，不代表推理运算量。

## 2. 必须公开的论文歧义 / 实现假设（不能伪装成作者源码）

1. **Table III 的通道笔误**：Encoder 第三组标记 `(16,32,4)`，但前一个 Pool 输出 `[B,32,26]`，按通道连续性必须用 `(32,32,4)`。
2. **Table III 的 FC 笔误**：先有 `(352,50)`，第二层又标 `(352,6)`；串联层应为 `(50,6)`，采用 `Linear(352,50)→LeakyReLU→Linear(50,6)`。中间 LeakyReLU 源文表格未标注，属合理实现选项。
3. **解码器输出长度需要额外填充**：Table III 指定卷积核 `(3,3,15,15)` 和四个特定输出长度，却没有 padding 细节；本复现使用 Conv1d 对称 zero-padding `(3,3,15,7)`，得到论文列出的每一层张量形状。这些 padding 是**根据表格反推**，不是可核对的原作者设置。保留了 Conv1d、BN、LeakyReLU、最近邻×2 Upsample。
4. **Loss1 的尺度歧义**：式 (11) 写作 `1/N Σ_i ||x_i−x̂_i||²₂`，严格相当于每样本 640 维平方误差**求和**，而常见框架 MSELoss 默认按 640 元素取**均值**。默认 `--reconstruction paper_sum` 严格按式 (11)，另允许 `element_mean` 作为**明确命名的敏感性实验**。两者相差 640 倍，不可无声切换。若 `paper_sum` 重构梯度占主导，应如实记录损失量级，不可凭目标域分数调整。
5. **DML 样本集合/类别不齐**：式 (14)–(18) 定义每类特征均值，却未写明 mini-batch 缺类的处理。本实现以本 batch 中实际出现的类数替代 C；无某类时不虚构其 centroid；只有一类时 inter=0。对全部六类在 batch 同时出现的情况，与原公式一致；这一选择属于实现假设。
6. **联合损失权重**：式 (20) 展示三个项均参与 encoder 梯度，没有显式权重系数，默认 `CE+1×recon+1×(intra−inter)`；`--rec-weight` 与 `--dml-weight` 仅供源域预先定义的敏感性试验，不以 Target-Dev 选择。
7. **优化器不完整**：论文描述 Adam、权重衰减、学习率衰减、初始 lr=0.001、batch16、50 iterations/rounds，但未给出数值 decay、scheduler 规则、随机种子等。本实现 `paper_like` 默认 Adam/WD=1e−4, StepLR(20,0.5) 是**明确的实现假设**，并非原文超参。验证选择“最高 clean Macro-F1→最低 clean 总损失→早 epoch”，也是原文“最低loss与最高accuracy”的可重复操作化，而非原作者明确的打分公式。
8. **原文“equidistant quadratic/secondary sampling”**：Eq (8) 给出采样间隔，不明确二次多项式插值核。`paper_like` 使用项目中已冻结的 `prepare.preprocess` 生成的 `[B,5,128]`，而 `matched` 使用 `[B,5,4096]` 先线性插值的周期和可微 MDNormalization，二者都是对论文预处理的工程近似。**不应该称训练前的线性插值为原作者官方二次插值算法。**
9. **论文样本数量与我们的冻结数据划分不同**：原文实验收集 1152 组六类平衡样本，采用 7:3:2。当前项目已冻结 34/15/10 条仿真原始 MAT 分组切分（738/302个周期）、88 条实测 Dev，且噪声测试协议更严格。即使核心方法忠实，也**不得把结果直接宣称为论文 99.58% 的数值复现**。

## 3. 两套实验，不可混称

### Track A：`paper_like`（尽量贴近原论文算法）

- 使用 **完整** `XuCAE`：Conv encoder + Conv decoder + classifier + Eq11/13/18 DML；不使用 A′ 权重，不引入频谱模块。
- 输入来自当前已冻结 `[N,5,128]` 的原 MD 处理；只用仿真 Source-Train 的 clean 输入进行训练，batch=16，50 epochs，Adam/WD、StepLR 数值采用第2节列明的假设。
- 源域验证选模，仅独立 Target-Dev 报告。作者原始数据采样/窗口策略和训练噪声做法不够明确，因此此轨道叫 paper-like reimplementation，不称 strict exact reproduction。

### Track B：`matched`（用于本会议论文的科学比较）

统一采用原 V4/V5 的 source-only 鲁棒性协议，使不同方法**只改变网络/前置模块**：

| Mode | Full CAE decoder | Eq11 recon | Eq14–18 DML | 额外前置模块 |
|---|---|---|---|---|
| `xu_full` | ✓ | ✓ | ✓ | 无；原始论文算法的完整骨干 |
| `xu_full_lp3` | ✓ | ✓ | ✓ | LP3 |
| `xu_full_lp4` | ✓ | ✓ | ✓ | LP4 |
| `xu_full_nahsf` | ✓ | ✓ | ✓ | V5 NA-HSF 门控 |
| `xu_cae_classify` | ✓ | ✓ | ✗ | 无，DML 消融 |

- Source Train：34 recording，738 周期；验证：15 recording，302 周期。
- 三种子 42/43/44；batch32，50 epochs，Adam lr=0.001，无 WD/调度（使 V4/V5 协议相同）。每轮按旧协议加入 10/15/20/25/30/35/40 dB 整记录逐相 RMS Gaussian；clean/noisy 配对损失。
- **扩展的论文目标**：在完全无实测训练数据前提下，比较 full Xu baseline 与 full Xu+LP3/LP4/NA-HSF 对**实验域未见的噪声强度**，特别是 0/5 dB synthetic stress 的鲁棒性。原论文已有 source-only sim→real 泛化和一般噪声抗扰能力，**不能再把这两点当作本文首次提出**。
- Source Val 统一固定 Clean 与 0/5/10/20/30；Clean F1 保持最高值 1pp 以内，从中按五档平均 F1 最大选 epoch，再按 Clean 与早 epoch 排序；Source-Test 和 Final-Holdout 不参与选择。
- Target-Dev 88 记录、每条件 2474 周期：原 10–40 dB 含噪实测与 Clean，另 0/5 dB 是从作者原 10 dB 残差缩放得到的**synthetic**，不是独立实测。
- 各 arm CNN 同种子初值应保持一致（`XuCAE` 在 shaping gate 之前实例化）；训练噪声序列一致；新模型 **不复用** A′ 25,620 参数，也不迁移先前 V5 权重。
- 除模型结构外，DML 和重构的默认权重、归一化约定对所有 matched arm 必须完全一致。

## 4. 关键测试与诊断

- 表 III 每层形状、Conv/Decoder/FC 参数对齐；检验 DML Eq14–18 的合成可解析数值，以及梯度同时流入 encoder/decoder/classifier。
- 形状 `[B,5,4096]→[B,5,128]→[B,352]→[B,5,128]+[B,6]`，全参数随机初始，source-only。
- 每种子保留 train CE/recon/intra/inter、源域验证分数、selected epoch，以及源域配置审计；输出宏F1、逐类 F1/Recall、混淆矩阵、健康误报、故障→健康。
- **损失量级检查必须先于正式50轮**：若严格 Eq11 重构项数量级明显高于 CE，可另做 `element_mean` 明示实验；不可以 Target-Dev 结果选择隐含权重。
- 复现期不得读取 347 Final-Holdout。保留源域 Test 10 records 不用于选模。
- 本代码暂不执行或声称与作者公布的 99.58% 相同的结果，也不声称修改的 NA-HSF 在完整 CAE+DML 骨干上会继续有 V5 的收益。

## 5. 服务器实际运行方法

将分支 checkout 到原项目根目录，使用原 Python/uv 环境：

```bash
python scripts/xu2025_full_model.py
python scripts/xu2025_full_experiment.py smoke
python -m pytest -q tests/test_xu2025_full_model.py tests/test_xu2025_full_experiment.py

# A: 完整原论文 CAE+DML（原文未披露参数由文档明确约定）
python scripts/xu2025_full_experiment.py train --track paper_like --modes xu_full \
  --out outputs/xu2025_full

# B: 同训练和五档验证协议下，四种完整 Xu CAE+DML 结构对照
python scripts/xu2025_full_experiment.py train --track matched \
  --modes xu_full xu_full_lp3 xu_full_lp4 xu_full_nahsf \
  --source-strong-root '/path/to/generated/source' \
  --out outputs/xu2025_full

# B: 含作者原始10-40dB及合成0/5dB的完整88-record Dev
python scripts/xu2025_full_experiment.py evaluate --track matched \
  --modes xu_full xu_full_lp3 xu_full_lp4 xu_full_nahsf \
  --snr-root '/path/to/original/SNR-parent' \
  --strong-root '/path/to/generated/target' --out outputs/xu2025_full

python scripts/xu2025_full_experiment.py summarize --track matched \
  --modes xu_full xu_full_lp3 xu_full_lp4 xu_full_nahsf \
  --out outputs/xu2025_full
```

只训练一次每种子 50 epoch，禁止覆盖 checkpoint。训练/评估/结果保存按 `paper_like` 与 `matched` 独立目录隔离，模型名和损失配置一并写入元数据。若服务器发现任何依赖/输入 shape 问题，先修复并重跑 smoke/test，严格记录 commit，再开始正式训练。

**学术结论边界**：此次工作建立的是可核对论文依据的完整 CAE+DML 实验基础。会议论文的新贡献必须由 Full CAE+DML matched baseline 和 Full+NA-HSF 的独立实测决定，不能借 A′ 的历史结果替代完整原论文对照，也不能隐瞒原论文已有的高性能或已有域泛化贡献。
