# HUST-SNR V3：原论文 CAE-Classification + 低通先验可学习滤波

**状态（2026-10-12）：运行提交`d483866`已完成四方法×三种子×50轮GPU训练，以及Clean＋七档SNR共96次Dev评估。29项测试通过，48组源Val分数与96组Dev预测复现一致。Final-Holdout未使用。** [完整服务器记录](../records/v3_cae_prior/server_d483866/SERVER_RESULTS.md)。

## 1. 原论文结构核对与重要区别

源论文：Haoxiang Xu 等，*Fully Simulated Data-Driven Domain Generalized Method for Multiphase Converters Fault Diagnosis*，IEEE TPEL 40(1), 2099–2113 (2025), DOI [10.1109/TPEL.2024.3461159](https://doi.org/10.1109/TPEL.2024.3461159)，**Fig.4、Table III、式(9)–(13)**。

原论文 `CAE-Classification` 是 Encoder + Decoder（重构 MSE，Loss1）+ 分类头（CE，Loss2）；**它并不包含作者进一步提出的深度度量学习 Loss3**。与此不同，原论文最终提出的方法称 `CAE-Classification+Metric`，需要另加 Loss3。

现有仓库 V1/V2 `OriginalCNN` **已经复制了原论文 Encoder＋Classifier 的卷积/池化/352→50→6 分类结构**，不是任意替代网络。因此 V3 增加的核心是 Decoder 及训练时的重构 MSE，然后考察是否能与频域滤波互补。

原论文 Table III 有内部维度矛盾：第三层Encoder的输入通道印为16，但前层输出为32；分类层第二个Linear印为352→6，与第一层352→50衔接不符；Decoder列出的若干卷积核和输出长度也不能由普通Conv1d算子同时得到。V3保留明确的Encoder/Classifier实现，并以**转置卷积+显式两端各1点填充**实现Decoder，使其可运行并匹配作者给出的长度。**这是声明过的近似结构复现，而非作者源码的精确再现。**

## 2. 借鉴 FilterNet 及先验初始化

另一来源：Yi 等，*FilterNet: Harnessing Frequency Filters for Time Series Forecasting*, NeurIPS 2024，[官方 PaiFilter](https://github.com/aikunyi/FilterNet/blob/main/models/PaiFilter.py)。原 PaiFilter 用共享可学习时域卷积核经 rFFT 相乘；本项目使用**五相共享实数频谱增益**，是任务适配，不是原 PaiFilter 的逐字复现。

V2结果：`lowpass3+CNN` 在10dB Dev Macro-F1 = **95.79%**，但V2默认近恒等初始化可学习增益大都在0.91–1.00，不能有效抑制高频。因此V3从三阶低通**近似状态**初始化，而不是从恒等滤波初始化：

\[
 g_0=1,\qquad
 g_k=\sigma\left(b_k+8\Delta_k\right),\quad
 b_{1,2,3}=+4.6,\quad b_{k\ge4}=-4.6.
\]

初始化前三阶增益约0.9900，高阶约0.0100。共128个可学习参数：前32阶独立，其余96个频率控制点线性插值覆盖高阶，五相共享；增益均在(0,1)，不引入相位旋转，既可增强某个高阶增益，也可降低当前低阶增益。所有滤波在原始电流**MD极值归一化之前**，保留低电流相除100规则。

## 3. 算法流程和训练目标

```text
五相原始物理电流 → 冻结完整电周期 → 4096点采样 → FFT × 滤波增益 → IFFT
   → HUST原MD归一化 → 5×128
   → Encoder（原CNN卷积层）→ [32×11] ─→ 分类头352→50→6 → CE
                             └→ Decoder→重构5×128 → MSE（训练专用）
```

对源域干净与Gaussian噪声视图分别计算CE及MSE，固定 `lambda_rec=1.0`，重构的是各视图**自身经过频域整形和MD后的输入**，不是“噪声输入重构Clean”。因此本模型称**CAE分类**，不声称是监督去噪自编码器。重构目标`detach`防止自我移动目标的梯度直接流回输入，Decoder不参与推理。

## 4. 四组公平对照

| 名称 | 输入频谱处理 | CAE Decoder + MSE | 目的 |
|---|---|---|---|
| `cae_identity` | 恒等、不滤波 | 是 | CAE 原结构基线 |
| `cae_lowpass3` | 固定 DC、前3阶；其余置零 | 是 | CAE＋传统低通 |
| `cae_prior` | **低通先验初始化＋可学习频谱增益** | 是 | V3候选主方法 |
| `cae_prior_no_rec` | 同上 | **否** | 区分Decoder重构训练是否真正有益 |

参数：原Encoder＋分类器25,620；Decoder9,845（仅训练）；新增可学习频域参数128；`cae_prior`训练总参数35,593，**推理所需可训练参数25,748**（不运行Decoder）。

另外保留已完成V2的`lowpass3+CNN`为独立的历史强基线（10dB 95.79% Dev Macro-F1）。若`cae_lowpass3`没有超过它，不能说CAE重构有益；若`cae_prior`没有超过`cae_lowpass3`，不能说低通先验可学习滤波强于传统滤波。

## 5. 冻结实验协议与代码复用

V3的训练、原MAT测试、数据划分、采样、噪声RNG、源验证Checkpoint规则**直接复用**`scripts/v2_experiment.py`，只有模型类型、训练损失和独立输出目录通过`scripts/v3_experiment.py`替换。这样避免新增另一套数据流程。

- Source-Train:34条原记录/738个周期；Source-Val:15/302，严格模拟源域；
- seeds:42/43/44；Adam lr=0.001，batch32，epochs50，双视图每个CE占0.5，额外重构MSE权重1；
- 选模：仅源域Clean性能距最优不超过1pp，选10/20/30dB固定源域带噪验证Macro-F1均值最高的epoch；
- Target-Dev:88条独立原记录/2474周期；Clean＋10/15/20/25/30/35/40dB仅用于评价，不用于参数更新或checkpoint选择；
- **Final-Holdout 347条不读取**；同样禁止根据这次Dev结果无限次调整先验。
- 论文中的所有最终对比应报告三个种子均值/标准差及配对增益，清楚区分源域验证和目标Dev。原作者的Accuracy和本项目Macro-F1及划分不同，不可直接混作同指标。

## 6. 在GPU服务器执行

```bash
git fetch origin
git switch research/cae-lowpass-prior-v3

python scripts/v3_cae_prior.py
python scripts/v3_experiment.py smoke
python -m pytest -q tests/test_v3_cae_prior.py tests/test_v3_experiment.py

# 三种子 × 四方法，每方法50 epoch
python scripts/v3_experiment.py train

# 使用原MAT，--snr-root指向包含SNR_10等文件夹的父目录
python scripts/v3_experiment.py evaluate --snr-root "/PATH/TO/SNR_PARENT"

python scripts/v3_experiment.py summarize
```

或提供先前导出的88条Dev资料：`evaluate --dev-export-root /path/to/dev_snr_export`；报告正式指标优先使用原MAT。

全部结果存于 `outputs/v3_cae_prior/`，不修改Gate0/V1/V2既有模型与预测；模型、历史、选模JSON按方法/种子分开保存，目标Dev逐周期预测/混淆矩阵保留，汇总见`SUMMARY.json`及`PER_SEED_RESULTS.csv`。重复运行同一训练输出会被V2原代码拒绝覆盖。

本轮实际使用独立目录`/home/wang/projects/会议论文/hust/experiments/v3_cae_prior/d483866/results/`，四组并发。正式训练、测评及汇总墙钟为56.02秒，不含部署、预检和独立核验；未修改模型代码、超参数或原环境。训练history继承V2，只保存总损失；最佳权重Source-Val eval模式下的CE/MSE分量另存核验JSON。best.pt的train_protocol字符串继承V2而写为paired-source-CE，实际V3目标以本文件和代码为准：CE＋自身重构MSE，no_rec组MSE为0。

## 7. 不应过度解读的事项

- 新Decoder的边缘长度对齐是**实现选择**；不能将其归于Xu等的确切实现。
- 重构自身会重构剩余噪声，未必改善抗噪泛化；测试是探索，不代表必然优于CNN。
- 低通先验仅根据先前V2的Dev发现设定，当前研究阶段已有目标域分析信息；严格表述为**源域权重训练与源域checkpoint选择**，不是完全没有目标域参与方法开发。
- 低通频率门控只有128参数，在V2源域很快100%时也可能没有足够梯度改进；结果若不提高，应接受负面结论。
