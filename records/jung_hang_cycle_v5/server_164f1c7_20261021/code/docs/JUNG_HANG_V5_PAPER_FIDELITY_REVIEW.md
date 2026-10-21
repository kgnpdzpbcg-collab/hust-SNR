# Jung / Hang 高保真复现 V5：论文依据、实现差异与运行条件

> 代码已提交到独立分支 `research/jung-hang-paper-fidelity-v5`。本文件是**五相任务的论文机制导向复现**，不是作者官方源码，也不是已完成的真实数据训练报告。V2/V3/V4 权重与历史预测保持不变。

## 1. 一手材料及文中位置

- **Jung, Apsari, Lee (2025), IEEE TPEL**, DOI `10.1109/TPEL.2024.3488094`。关键证据：Sec. III-A / Fig. 7（两个电周期、单周期步长）；Sec. III-B / Fig. 9（空洞卷积、残差、GAP）；Sec. IV-A/B / Eqs. (1)–(5)（THD+N 白噪声、故障阻断区间限幅）；Sec. V-A（3%/5%/7%/10% THD+N、30 epochs、AdaMod）。
- **Hang, Shu, Ding, Huang (2023), IEEE TIE**, DOI `10.1109/TIE.2022.3231304`。关键证据：Sec. II-B / Eqs. (2)–(6)（Clarke 电流矢量，瞬时矢量模归一化）；Sec. II-C / Fig. 6 / Eqs. (10)–(12)（5 层卷积、2 次 Haar-DWT LL 池化）；Sec. III（15/20 dB AWGN 噪声鲁棒性实验）。
- 论文中网络卷积各层全部超参、图像栅格化规则、部分训练细节未完全披露。沿用已记录的 V3 结构选择时，要将其视为**实现假设**，不能称与原作者代码完全相同。

## 2. 修订前后的关键区别

| 项目 | V4 默认 | V5 默认（待服务器验证） | 依据或限制 |
|---|---|---|---|
| Jung 窗口 | 1 电周期 × 128 | 连续 **2 周期 × 128 = 256 点**，单周期滑动 | Fig. 7；固定重采样长度属五相任务适配 |
| Jung 双周期边界 | V4 双周期可选，首两窗可能重复 | **正向连续两周期**；缺邻居即丢弃样本；按 Train/Val/Target 独立构造 | 不跨分组、时间隔离或标签 |
| Jung 优化器 | Adam | **AdaMod** | Sec. V-A |
| Jung 训练轮数 | 50 | **30** | Sec. V-A |
| Jung 噪声 | 整记录 SNR=10–40 dB 的 AWGN | **THD+N=3%、5%、7%、10%**，依公式估计新增白噪声 RMS | Eqs. (1)–(5)，与 SNR 不等价 |
| Jung 阻断区间 | 默认全部注噪 | 按五相故障标签只对故障相的持续低电流区间限制噪声 | 原论文只针对 3L-NPC Sj2/Sj3，现有代码是必要**物理启发式适配**，不可称开关态等价 |
| Hang 变换 | 两个五相 VSD 平面 | **一个基波 VSD 平面** | 3 相 Clarke → 5 相替代；须核验物理相序 |
| Hang 归一化 | 全窗口共享尺度 | **逐采样时刻 `(alpha,beta)/sqrt(alpha²+beta²)`** | 原文 Eq. (6) |
| Hang 轨迹 | 2×40×40 栅格 | **1×40×40**，默认仅跳过精确零矢量（`hang_quiet_ratio=0`），允许另行测试噪声阈值 | 单轨迹对应原方法；40×40 采样及栅格线属于公开信息不足部分 |
| Hang CNN | 5 Conv + 2 Haar LL pooling | **保持**，输入通道改回 1 | Fig. 6 |
| Hang 增强 | 10–40 dB 配对 | 15/20 dB 白噪声与 Clean 配对 | 对齐原文抗噪实验中给出的噪声设置；配对批次细节是适配选择 |
| 评价样本 | 2474 / SNR / 方法 | Jung 删除分组末尾缺第二周期的锚点；**Hang 评价限制在同一锚点集合** | 避免两个方法用不同标签样本比较 |

## 3. 五相适配与不能做到完全原样复现的部分

1. **Jung 原故障对象为三相三电平 NPC，12 类开关故障和专门设计的不同触发时刻瞬态。** HUST 五相 IM-AA 的六类故障与拓扑并不等价；V5 不虚构未公开的故障触发时刻，也不将五相零电流门控称为 NPC 原限幅规则。保留源域稳态周期训练，明示为任务适配。
2. **THD+N 分布依赖基波 RMS 与原信号 THD。** HUST 仿真记录包括故障前后片段；V5 在有 `0.10–0.45 s` 健康片段时，从该段估计标称原生 THD 和基波 RMS，再按公式增加噪声。超出原生 THD 的目标才产生新增噪声：`sigma = I1 * sqrt(max(THDN_target²-THD_original²,0))`。预检报告 `thdn_nonzero_injection_fraction_by_target_percent`。若大多数记录注噪为零，应先检查频率估计及训练有效性，不能将训练完成误称有效增强。
3. **Hang 的五相基波 VSD 并非原始三相 Clarke。** 默认电角度 `[0,72,144,216,288]` **仍是待实测相序核验的模型假设**。预检统计健康段第三平面与基波平面的 RMS 能量比；真实相序应对照数据列物理含义核查，不可仅凭合成正弦测试推断正确。
4. Jung 仍采用从 V3 继承的紧凑残差 CNN（5→32、dilated 3、MaxPool、2 Conv 残差、GAP、FC），因为原论文未完整公开精确逐层维度；Hang 继承 V3 的 5 Conv、2 Haar LL pooling，默认通道 `(4,8,16,32,64)`。它们属于依据图示的**模型重建**。
5. 复现准确率不能与原论文在其他机器、故障类别、噪声强度和原始数据划分条件下报告的指标直接做数值复现比较。

## 4. 实验纪律和结果口径

- 只用既定 `Source-Cycle V2` 仿真源域训练和选模，仍是 **seed 42、44**。
- Jung / Hang 采用各自原论文重点机制及训练策略；不同架构不强制相同 epoch、优化器或噪声方式。
- 统一比较仅在相同 **Target-Dev 原始周期锚点**上计算指标。Jung 是锚点当前周期 + 下一完整周期，Hang 是锚点当前周期的图像。输出保存 `original_cycle_anchors` 和真实评价 `windows` 数量，**不再默认所有模型都是 2474 个周期**。
- Dev 用于复现检查，不用于 checkpoint 选择或按目标分数调参数；未读取 347 条 Final-Holdout。
- **尚未在 GPU 服务器跑 V5 的 source preflight、训练或 target evaluation。** Github Actions 的 CPU 测试只证明合成数据执行与基本不变量，并不证明真实输入正确或诊断性能已经改善。

## 5. 代码与推荐执行顺序

- `scripts/jung_hang_models_v5.py`：Jung 连续两周期与 THD+N、故障相低电流保护；Hang 单平面归一化轨迹
- `scripts/compare_jung_hang_v5.py`：模型输入、锚点与标签对齐
- `scripts/jung_hang_optim_v5.py`：AdaMod
- `scripts/jung_hang_cycle_v5.py`：预检、训练、测试、重放核验、结果汇总
- `scripts/jung_hang_v5_audit.py`：真实源域 THD 注噪覆盖率及 VSD 物理相序的替代诊断指标
- `configs/jung_hang_v5_v1.json`：冻结配置
- `tests/test_jung_hang_models_v5.py`：CPU 测试

服务器从当前仓库根目录切到 `research/jung-hang-paper-fidelity-v5`，沿用 V4 已验证的 PyTorch / 数据文件夹。提供以下四个已存在位置：
- `REFERENCE`：V4 / Source-Cycle 对照运行输出根目录，包含 `code/scripts` 和 `DEPLOYMENT_MANIFEST.json`
- `SOURCE`：Source-Cycle V2 数据根目录
- `ADDITIONAL`：4/−4/−8 dB 新增噪声数据归档根目录
- `PREVIOUS`：已有十二模型七档输出根目录（含 `results/reports/PER_SEED_RESULTS.csv`）

同一新建的 `OUTPUT` 中依次运行：

```bash
python scripts/jung_hang_cycle_v5.py preflight --reference-root "$REFERENCE" --source-root "$SOURCE" --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT"

for method in jung hang; do
  for seed in 42 44; do
    python scripts/jung_hang_cycle_v5.py train --method "$method" --seed "$seed" --reference-root "$REFERENCE" --source-root "$SOURCE" --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT"
  done
done

python scripts/jung_hang_cycle_v5.py evaluate --reference-root "$REFERENCE" --source-root "$SOURCE" --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT"
python scripts/jung_hang_cycle_v5.py verify   --reference-root "$REFERENCE" --source-root "$SOURCE" --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT"
python scripts/jung_hang_cycle_v5.py summarize --reference-root "$REFERENCE" --source-root "$SOURCE" --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT"
```

**训练前必须先查看** `OUTPUT/PREFLIGHT_PAPER_AUDIT.json`：每个 THD+N 档位有效注噪比例，以及健康电流 VSD 平面能量比。若比例不合理或物理相序不明，先停在 preflight，不应匆忙生成训练结果。

本分支不修改论文主算法 NCPHR-Net；也不覆盖历史 V4 权重或评价记录。
