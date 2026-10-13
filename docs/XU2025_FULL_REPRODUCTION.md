# Xu et al. (TPEL 2025)：完整 CAE + DML 单模型基线复现 V1

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

Decoder 对应 Table III：`[B,32,11] → upsample [B,32,22] → conv [B,32,26] → upsample [B,32,52] → conv [B,16,56] → upsample [B,16,112] → conv [B,16,128] → conv [B,5,128]`。分类器 `352→50→6`。复现网络总参数以实际运行 `parameter_counts()` 为准：**35,475 个可训练参数**。本轮仅评估原论文完整 CAE＋DML（Xu-Full），不训练任何“Full+模块”模型。此处参数量不代表推理运算量。

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

### Track B：`matched`（本次唯一需要执行的正式实验）

**只运行一个模型类型：`xu_full`，即 Xu 等人完整的编码器、解码器、分类器与 DML 联合训练方法。** 此 track 只是为了与已经完成的 A′／V5 实验使用相同的数据、训练噪声和 checkpoint 选择条件，**不是**原论文作者原始训练协议的完全复制。

- Source Train：34 条 recording，738 周期；Source Val：15 条 recording，302 周期。
- 三个随机种子 42/43/44，用于和既有 V5 三种子比较同一模型的重复性；每个 seed 训练一次 `xu_full`，不训练任何其它网络类型。
- 50 epochs、batch32、Adam lr=0.001、weight_decay=0；源域训练使用 10/15/20/25/30/35/40 dB 的逐相噪声增强及 clean/noisy 配对损失。**这是本项目统一对照协议，不能描述为原论文作者所用噪声训练。**
- 源域验证 Clean、0/5/10/20/30 dB；Clean F1 距最高分 1pp 以内时选五档平均 F1 最大的 epoch。同样，这属于本项目为消除源域验证饱和而固定的选模规则。
- Target-Dev：88 条 recording、每条件 2474 周期；Clean 和作者发布的 10–40 dB，另有合成的 0/5 dB 压力测试。绝不使用目标域进行训练、微调或选 checkpoint；Final-Holdout 仍隔离。
- 历史 A′、A′＋LP3、A′＋LP4、固定 α=0.5、V5 NA-HSF 的模型、预测和记录均保持不变，不再新增完整 Xu＋频域模块的模型臂。
- **研究用途**：将 Xu-Full 作为论文所引用工作的完整方法基线；对比同一任务与数据协议下的诊断鲁棒性和仿真→实测迁移表现。由于 Xu-Full 与 V5 同时在 backbone 和训练目标上不同，二者差值属于**整体方法比较**，不能单独归因为 NA-HSF 模块。模块贡献应在既有 A′ 系列消融中讨论。

`paper_like`（仅原 Xu-Full、偏向原作者公开训练设置）保留为可选的更贴近论文设置的复现实验，用于单独报告重现性；**不是本轮服务器一键脚本需要运行的第二个训练任务**。

## 4. 关键测试与诊断

- 表 III 每层形状、Conv/Decoder/FC 参数对齐；检验 DML Eq14–18 的合成可解析数值，以及梯度同时流入 encoder/decoder/classifier。
- 形状 `[B,5,4096]→[B,5,128]→[B,352]→[B,5,128]+[B,6]`，全参数随机初始，source-only。
- 每种子保留 train CE/recon/intra/inter、源域验证分数、selected epoch，以及源域配置审计；输出宏F1、逐类 F1/Recall、混淆矩阵、健康误报、故障→健康。
- **损失量级检查必须先于正式50轮**：若严格 Eq11 重构项数量级明显高于 CE，可另做 `element_mean` 明示实验；不可以 Target-Dev 结果选择隐含权重。
- 复现期不得读取 347 Final-Holdout。保留源域 Test 10 records 不用于选模。
- 本代码暂不执行或声称与作者公布的 99.58% 相同的结果，也不声称修改的 NA-HSF 在完整 CAE+DML 骨干上会继续有 V5 的收益。

## 5. 服务器实际运行方法：只跑 Xu-Full

从仓库根目录运行，在原 V5 已验证的 Python/CUDA 环境中指定数据目录：

```bash
git fetch origin
git switch research/xu2025-full-cae-dml
export SOURCE_STRONG_ROOT="/path/to/generated/source"
export TARGET_SNR_ROOT="/path/to/original/target"
export TARGET_STRONG_ROOT="/path/to/generated/target"
bash scripts/run_xu2025_full_server.sh
```

该脚本依次执行 smoke/tests、**唯一模型 `xu_full` 在 `matched` 轨道下 42/43/44 三种子训练**、对应 Clean/10–40/5/0 dB 的 Target-Dev 评估与汇总。每种子 50 epochs；不触碰 Target-Final-Holdout，也不覆盖已有 checkpoint。

执行完后读取 `outputs/xu2025_full/matched/reports/SUMMARY.json`，并使用已经保存的 V5 六种子与三种子结果作清晰标注的整体方法比较。只有统计口径相同、随机种子相同的子集可以进行配对比较。

**复现状态**：目前仅实现并通过本地模型测试；服务器正式 `xu_full` GPU 结果尚不存在。原论文中的填充、联合损失尺度、mini-batch DML 规则和部分优化器超参数并未完全披露，上述差异与实现假设已在第2节标注。若 Xu-Full 结果优于现有 V5，需如实报告，不能预先假定原论文方法存在性能缺陷。若 Xu-Full 较差，也只能说明在本项目冻结协议下该实现存在局限，不能将其无条件外推为原作者报告方案的普遍缺陷。
