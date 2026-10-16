# NCPHR-Net (B1) 会议论文消融实验 V1 — 代码实施与核验协议

> **状态：代码已实现并提交；尚未在真实 HUST 源训练数据上完成 A1–A5 训练，所有新增消融成绩仍为 TBD。**
> 
> 基线：当前仓库冻结的 [B1 实现](../scripts/hust_b1_phase_harmonic.py)；论文故事 [PAPER_STORY_NCPHR_METHOD_ABLATION_V1.md](PAPER_STORY_NCPHR_METHOD_ABLATION_V1.md)。研究分支 \`research/b1-paper-ablations-v1\`，与原 \`research/a1-b1-noise-structure-v1\` 实验完全隔离。
>
> **目标**：从 Introduction 两项研究贡献反向确定对照变量，不是为了堆积消融数量而任意删除网络层。

## 1. 五个消融的精确定义

| 方法 ID | 对应方法贡献 | 唯一/主要修改 | 不修改哪些结构 | 可支持的论文结论 | 不能支持的结论 |
|---|---|---|---|---|---|
| \`a1_fixed_lp4_dual\` | 贡献1：学习式恢复 | k0..4 固定保留，k>4 恒为零，删除可学习恢复器及其预训练/辅助损失 | 保留两条五相视图、10 通道 ResNet、配对 CE | 学习式恢复**完整子系统**的增量价值 | 不能单独证明哪个噪声代理或哪项损失有效 |
| \`a2_fixed_noise_proxy\` | 贡献1：按输入噪声调整 | 每条样本和相的频谱高阶噪声代理改成**仅从 Source-Train 统计得到的固定标量** | 保留 k0..4、逐阶噪声校准参数、所有跨相描述、全部恢复损失、双视图 | **样本级噪声统计**的增量价值 | 不能声称“去掉了整个噪声校准机制” |
| \`a3_no_interphase_descriptors\` | 贡献1：跨相描述信息参与恢复 | **只**将恢复增益预测器第 5:15 个输入通道（相对功率、Gram 相干）置零 | 原 Conv 参数量不变；共享基波尺度、噪声代理、强度损失、双视图均保留 | 两组跨相描述符对**收缩决策**的价值 | 不能声称去掉了系统中一切跨相信息 |
| \`a4_md_only\` | 贡献2：共享相间幅值视图 | 去掉第二个五相共享尺度幅值视图，分类器 stem 从10输入改成5输入 | 全部恢复器、损失、源预训、ResNet 残差层 | 共享幅值视图在 MD 之外的**额外价值** | 不能仅凭一组消融证明两个视图均不可替代（可补 Amplitude-only） |
| \`a5_no_strength_loss\` | 贡献2：相间强度保持监督 | 仅把 \`phase-strength\` loss 系数从0.4改成0.0，**预训练+联合训练两阶段一致** | 完整 B1 推理图、噪声估计、双视图、其他损失不变 | 额外强度监督对最终诊断的作用 | 不能声称“移除所有故障物理结构” |

**原始 Full B1 不重新包装、修改代码或移动 checkpoint**；从之前统一基准表中读取相同 Source-Cycle V2、seed42/44 的 Full B1 行，与新增消融结果配对对照。

## 2. 实际文件

| 文件 | 用途 |
|---|---|
| [\`scripts/b1_paper_ablation_models.py\`](../scripts/b1_paper_ablation_models.py) | 单一变体注册表、五个隔离模型、源训练常量噪声代理标定、损失和恢复器预训练 |
| [\`scripts/b1_paper_ablations.py\`](../scripts/b1_paper_ablations.py) | 独立 \`train / evaluate / summarize\` 命令，沿用统一基准的循环、数据指纹、选模、权重保存与重新加载核对 |
| [\`configs/b1_paper_ablations_v1.json\`](../configs/b1_paper_ablations_v1.json) | 冻结五模型、两个种子、SNR 范围及源数据版本 SHA |
| [\`tests/test_b1_paper_ablations.py\`](../tests/test_b1_paper_ablations.py) | 模型输出、梯度、频谱精确投影、屏蔽位置、A2 buffer、A5 损失和协议检查 |

五个变体**不需要复制五套 B1 代码**。通过复用 \`B1PhaseAwareRecoveryNet\` 与 \`PhaseAwareHarmonicRestorer\` 实现，真正的干预尽可能局限在对应行为上。

## 3. 关键代码细节

### A1：固定 LP4 + 原双视图分类器

- \`FixedLP4Restorer\` 只做正交 rFFT → 保留 k=0..4 → 零填充 → irFFT，不含任何参数。
- 保留 \`B1PhaseAwareRecoveryNet.forward\` 中 **MD＋共享尺度视图** 以及后续原 ResNet。
- 原模型的 \`paired_loss\` 中 A1 使用 Clean/Noisy 各半权重 CE，不计算恢复损失。
- 不进行 10 轮无意义的恢复器预训练。这个消融**同时失去学习式恢复和其监督训练**，所以它是子系统级对照而非单算子消融。

### A2：固定噪声代理来源及恢复

- **只用源域 Train**，在该种子下使用冻结的源训练噪声分档与第 1 轮独立增强规则，采集训练样本每相 k20..80 的归一化功率中位数。
- 汇总为所有 Source-Train 样本、相的单个 robust median 标量 \(\rho_{\mathrm{fixed}}\)，存为 \`restorer.fixed_proxy\` 的 persistent buffer；这一步在预训练前执行。
- 恢复器内部对 \(\rho_p(x)\) 的所有使用（噪声方差与 \`adjustment\` 特征）均由 \(\rho_{\mathrm{fixed}}\) 替代，而不是只改一条支路。
- 源训练和实测推理都使用**同一个已经保存**的源标定常数，不使用目标样本统计进行固定标定。虽然原样本信号功率仍参与 SNR 类特征，但原来**随输入变化的噪声代理**不再存在。
- 未标定的 A2 在 forward 会显式报错；\`best.pt/last.pt\` 的 \`state_dict\` 持久化固定标量，重新加载采用 \`strict=True\`。

### A3：只改调节器输入

原 B1 \`adjustment\` 输入是 \`[B,26,5]\`，按 5 相依次：

\`\`\`text
 0:5   log1p(signal_power)          KEEP
 5:10  log1p(relative_power)       MASK
10:15  phase_consistency / Gram    MASK
15:20  log1p(noise_proxy)          KEEP
20:25  tanh(log_snr/8)             KEEP
25     harmonic order             KEEP
\`\`\`

通过不可训练的 \`MaskInterphaseDescriptors\` 紧挨原 \`adjustment\` 前插入，**原始卷积层、权重维度和总可学习参数数量一致**。参数梯度不接收被屏蔽通道上的数据。模型重新实例化后屏蔽仍存在，避免推理恢复时漏用。

### A4：只更改分类端输入视图

- B1 恢复器完整保留，仍输出五相重构波形；
- \`MDNormalization\` 的五通道视图保留；
- 仅去掉 \`_physical_view\`，将 ResNet stem 通道数设为 5；后续 ResNet 残差块完全不变；
- \`return_recovery=True\` 接口仍返回四元组，兼容原恢复损失；
- 与 Full B1 需要**从头独立训练**，不能直接借原 10 通道 stem 权重删除五通道。

### A5：只改 loss 而不改推理

仍使用原 \`B1PhaseAwareRecoveryNet\` 的前向网络（该类只作空子类）；在**恢复器预训练**和**联合训练**的 \`recovery_loss()\` 中设 \`w_phase=0\`，其余 \`complex/noise\` 权重完全不变。与 Full B1 参数量相同，推理结构完全相同。

## 4. 统一冻结训练协议

- 仅用已验证 Source-Cycle V2 源域：63 个仿真记录、962 Train、221 Val；每周期五相同步电流 \`4096\` 点。
- 两个指定种子：**42、44**，没有其他随机种子。
- 每一轮同样配对 Clean 与随机 SNR 10/15/20/25/30/35/40 dB 源数据，原始 source-validation Clean + SNR 0/5/10/20/30 dB。
- 每个模型同一 Adam lr=0.001、wd0、batch32、联合训练 50 epoch；A2–A5 额外 10 轮源恢复器预训练，A1 不需要。
- checkpoint 仍只依据**源验证**：Clean F1 距最大1百分点以内 → 五档源噪声平均 F1 最优 → Clean F1 → 平均源 CE 最低 → epoch最早。
- 提取被选择轮次权重的完整 \`state_dict\` 包括 BN 和 A2 标定 buffer，保存 best/last、history、选择记录、数据 SHA 与第一轮训练噪声哈希。
- 只在**上述步骤完成后**，对冻结的同一 Target-Dev 88 家族、每 SNR 2474 周期进行 5/0/-5/-10 dB 推理。此次不使用 Final-Holdout，不用实测标签决定新模型的 epoch/损失超参数。
- 无需额外改变数据源、划分、故障标签或深度学习分类骨干。

## 5. 服务器使用步骤

以下命令在服务器上仓库根目录执行，参数路径按实际服务器已有数据目录填写：

\`\`\`bash
# 1. 安装仓库原有环境，先运行自动化单元/集成烟雾测试
python -m pytest -q tests/test_b1_paper_ablations.py

# 2. 训练五个消融 x Seed 42/44（10组），结果写入全新隔离目录
python scripts/b1_paper_ablations.py train \
  --methods a1_fixed_lp4_dual a2_fixed_noise_proxy \
            a3_no_interphase_descriptors a4_md_only a5_no_strength_loss \
  --seeds 42 44 \
  --source-root /ABS/PATH/source_cycle_split_v2 \
  --out /ABS/PATH/b1_paper_ablations_v1

# 3. 仅评估固定源域 best 权重，读取实测 Dev 的四档 SNR
python scripts/b1_paper_ablations.py evaluate \
  --methods a1_fixed_lp4_dual a2_fixed_noise_proxy \
            a3_no_interphase_descriptors a4_md_only a5_no_strength_loss \
  --seeds 42 44 \
  --source-root /ABS/PATH/source_cycle_split_v2 \
  --strong-root /ABS/PATH/strong_noise_target \
  --extreme-root /ABS/PATH/extreme_noise_target \
  --snr-root /ABS/PATH/original_author_target \
  --out /ABS/PATH/b1_paper_ablations_v1

# 4. 汇总。输出 CSV：逐种子、逐类以及两种子均值/样本标准差
python scripts/b1_paper_ablations.py summarize \
  --out /ABS/PATH/b1_paper_ablations_v1
\`\`\`

命令中的三个目标域目录沿用原 \`source_cycle_benchmark.py\` 的读取器；实际在 5/0 dB 取强噪声目录，-5/-10 dB 取极强噪声目录。源域目录必须与原基准的 manifest 指纹一致。代码中的防覆盖逻辑对已经存在的 \`models/{method}/seed{seed}\` 目录报错，不会静默覆盖权重；对已有实测评估文件同样拒绝覆盖。

**各方法独立训练可以分多次运行；只有 5 方法×2 种子×4 条件齐全后才能执行 summarize。**

## 6. 提交论文前如何判断消融是否支持贡献

论文主张与验证对象的对应：

| 主张 | 比较 | 要查看的结果 |
|---|---|---|
| 学习式恢复比固定 LP4 更好 | Full B1 vs A1 | 四档 Macro-F1；弱相与复合开路召回；注意 A1 同时取消恢复监督 |
| 输入相关噪声代理帮助跨噪声鲁棒 | Full B1 vs A2 | 不同 SNR 的 F1 退化及噪声标定行为；特别是 -5/-10 |
| 相间复谱描述符提供额外收益 | Full B1 vs A3 | 复杂开路故障 Recall，分 SNR 的 F1 |
| 两种视图比仅 MD 更好 | Full B1 vs A4 | 单开关及开相的 Recall；5/0与强噪声都需兼顾 |
| 相间强度辅助监督有益 | Full B1 vs A5 | 有/无 strength loss 的分类差异，不把 loss 降低自身等同于 F1 提升 |

强烈建议同时提供 **Seed42 vs Seed44 的配对差值**，再报告平均±样本SD。只能根据实验支持的主张写 Methods/Conclusion，不提前认定五项必然都有效。

## 7. 当前完成与未完成

- [x] 基于仓库原 B1 代码定义五个精确干预对象。
- [x] 五变体代码、共享训练评估 CLI、独立 config、自动化测试文件提交。
- [ ] 在服务器配置原数据和完整 Python 环境，运行 \`pytest\` 并记录真实测试日志。
- [ ] 在 Source-Cycle V2 下执行10组新源域训练、固定权重四档实测 Dev 评估。
- [ ] 导出 A1–A5 结果并与已归档 B1 seed42/44 结果合表、检查论文主张。

**本文件不是实验结果报告；A1–A5 无结果之前不得填造论文消融数字。**
