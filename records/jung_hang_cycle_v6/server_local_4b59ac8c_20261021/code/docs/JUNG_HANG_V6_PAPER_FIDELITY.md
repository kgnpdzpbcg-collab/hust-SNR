# Jung / Hang 高保真复现 V6：论文条款与实现对照

## 0. 状态

- 代码已完成：`scripts/jung_hang_models_v6.py`、`scripts/compare_jung_hang_v6.py`、`scripts/jung_hang_optim_v6.py`、`scripts/jung_hang_cycle_v6.py`、`configs/jung_hang_v6_v1.json`、`tests/test_jung_hang_models_v6.py`。
- 本地 CPU 测试 10/10 通过（本机无 pytest，用临时目录中的 `pytest` 替身直接调用全部 `test_*` 函数；服务器仍需用 pytest 正式跑一遍）。
- 本地已用真实 Source-Cycle V2 数据跑通 `preflight`，审计结果见第 4 节。
- 本地 CPU 已用 1 个 epoch 跑通 `train` 全流程（Jung/Hang 各一次，种子 42，仅验证可执行性，不是实验结果）。
- **GPU 训练、Dev 七档评价、重放核验、汇总尚未执行。**
- 未读取 347 条 Final-Holdout；目标数据不参与训练或选模。

## 1. 一手材料

- **Jung, Apsari, Lee (2025), IEEE TPEL**, vol. 40, no. 2, pp. 3553–3565, DOI `10.1109/TPEL.2024.3488094`。
  - Sec. III-A：录制长度为两个电周期，按单周期窗口切分，输出为 `3 × window size` 矩阵。
  - Sec. III-B：10 kHz 采样、50 Hz 基波时 **200 点**代表一个周期。
  - Sec. III-A：归一化把数据缩放到 [-1, 1]，保留极性。
  - Sec. IV：式 (1) THD 只含 2、3、4 次谐波；式 (5) `σ = I1·sqrt((THD+N)² − THD²)`。
  - Sec. IV-B：Sj2/Sj3 故障用限幅器阻断半周期噪声；Sj1/Sj4 只降幅值，不加限幅。
  - Sec. V-A：THD+N 取 3%/5%/7%/10%，每个模型 30 epochs，优化器 AdaMod。
- **Hang, Shu, Ding, Huang (2023), IEEE TIE**, vol. 70, no. 8, pp. 7653–7663, DOI `10.1109/TIE.2022.3231304`。
  - 式 (5)：`|Is| = sqrt(iα²+iβ²) = (3/2)Im`，Im 是该工况的定子电流幅值（标量常数）。
  - 式 (6)：`Isn = Is/|Is|`，有只含常数系数的闭合形式。
  - 原文明确：健康与故障轨迹 "have the same fixed radius"；单开关故障为不同半圆，双开关故障出现六分之一圆、三分之一圆和直线。
  - 网络：5 层卷积 + 2 次 Haar-DWT LL 池化，C7 输出 10×10×32。
  - 噪声实验：每相 20 dB、15 dB AWGN，含噪轨迹图与原样本混合。
- 两篇论文均未公开作者源码；逐层超参、栅格化规则、部分训练细节未完整披露。

## 2. V5 → V6 的修正

| # | 项目 | 原文依据 | V5 实现 | V6 实现 | 实测影响 |
|---|---|---|---|---|---|
| 1 | Jung 输入窗口 | Sec. III-A "two-cycle length … sliced with a one-cycle size window"；Sec. III-B "200 data points … 50 Hz at 10 kHz" | 2 周期 × 256 点 | **1 周期 × 200 点** | 源域样本由 884/143 恢复为 962/221 |
| 2 | Hang 归一化 | 式 (5) Im 为标量常数；"same fixed radius"；故障形态含直线 | 逐采样时刻除以自身模长 | **按记录健康段估计的标量 Im** | V5 轨迹全部落在单位圆上（比例恒为 1）；V6 第 0 通道比例为 0.771 |
| 3 | Jung THD 阶数 | 式 (1) 只含 2、3、4 次 | 累加到 30 次 | **只统计 2、3、4 次** | 与第 5 项合计把原生 THD 中位数从 2.10% 降到 0.058% |
| 4 | Jung 限幅范围 | Sec. IV-B 只对 Sj2/Sj3 限幅 | 按标签对故障相持续低电流区限幅 | **仅 OLS/OP（2/3/5 类）限幅；OUS（1/4 类）不限幅** | 见 `BLOCKED_PHASES` |
| 5 | 原生 THD 的 FFT 泄漏 | 式 (1) 要求谐波谱线可测 | 矩形窗直接做 FFT | **参考段裁到整数个电周期后再做 FFT** | V5 原生 THD 中位数 2.10%（30 次谐波 + 泄漏）、3% 档仅 63.5% 通道注噪；V6 0.058%（只统计 2/3/4 次 + 整数周期裁剪）、3% 档 100% 通道注噪 |
| 6 | Hang 输入平面 | 三相 Clarke 平面在五相下的推广 | 单基波平面 | **基波 + 三次双平面** | 见第 3 节 |

### 2.1 第 6 项的数据依据

五相电机无中线时有 4 个独立自由度，对应基波与三次两个 VSD 平面；三相电机只有 2 个自由度，Clarke 平面即完整状态。实测本数据集（962 个源域训练窗）：

- 基波平面各类平均图的 L1 距离：1/2/3/5 类之间仅 0.003–0.011，4 类与其余为 0.031–0.035。
- 三次平面各类平均图的 L1 距离：0.018–0.114。
- 单相开路后剩余四相仍能维持基波旋转磁动势：3 类（OP）A 相故障段 RMS 仅 0.02（健康为 0.707），但基波 αβ 模长仍为 1.00。
- 三次平面每窗最大模长（相对健康基波）：0 类恒为 0.00，1/2/3 类约 1.00–1.05，4 类最高 2.64，5 类最高 1.63。

因此单基波平面会丢掉主要的类间信息，V6 采用双平面。轨迹坐标共用同一尺度 `max(健康矢量模长, 本窗两平面最大模长)`：前者对应原文常数 Im，后者只在三次平面幅值超过健康基波幅值时生效，防止轨迹被栅格边界截断。

## 3. 五相适配与不能原样复现的部分

1. **Jung 的故障对象是三相三电平 NPC**（12 类开关故障、专门设计的触发时刻瞬态）。HUST 五相 IM-AA 的六类故障与拓扑不等价；V6 不虚构未公开的触发时刻，限幅规则按"相电流是否被完全阻断"映射到五相标签。
2. **Hang 的 Clarke 平面在三相下只有一个**；V6 用基波 + 三次双平面作为五相推广，网络首层输入通道由 1 改为 2（HangWCNN 参数 63 294 → 63 330）。
3. **Hang 的五相绕组空间相位 `[0,72,144,216,288]` 仍是模型假设**，尚未用数据列的物理含义核验相序。
4. **网络逐层超参属于依据图示的重建**：Jung 为 5→32 卷积 + dilated 3 + MaxPool + 2 残差卷积 + GAP + FC（7 110 参数）；Hang 为 5 层卷积 + 2 次 Haar-LL 池化（63 330 参数）。
5. **训练噪声口径**：Jung 只使用原文选定的 3% THD+N（V5 在 3/5/7/10% 间轮换）；Hang 使用 15/20 dB AWGN。两者每个 batch 同时前向 clean 与含噪视图，等价于原文"含噪样本与原样本混合"的 50/50 混合。
6. **评价不能与原文指标做数值对齐**：故障类别、数据划分、机器与噪声生成方式均不同。

## 4. 本地预检实测（真实 Source-Cycle V2，63 条记录）

`preflight` 输出（`PREFLIGHT_PAPER_AUDIT.json`）：

| 项目 | 数值 |
|---|---|
| 源划分 manifest sha256 | `4a0c76fd…eed3d`（与冻结值一致） |
| 源域 Train / Val | 962 / 221 |
| Jung 原生 THD（中位 / p95 / 最大） | 0.058% / 0.098% / 0.102% |
| Jung 3% 档实际注噪通道比例 | 1.000 |
| Hang 健康矢量模长（中位 / 最小 / 最大） | 4.786 / 4.547 / 6.335 |
| Hang 健康参考段回退记录数 | 0 |
| Hang 训练图：不同图像数 / 跨标签重复组 | 605 / 0 |
| Hang 训练图：通道 0 / 通道 1 落在单位圆比例 | 0.771 / 0.084 |
| Hang 验证图：不同图像数 / 跨标签重复组 | 165 / 0 |
| Hang 验证图：通道 0 / 通道 1 落在单位圆比例 | 0.774 / 0.084 |
| CPU 前向自检 | Jung (2,6)、Hang (2,6) |
| CPU 单元测试 | 10/10 通过 |
| CPU 1-epoch 训练冒烟（种子 42，源验证五档噪声均值 Macro-F1） | Jung 0.0478；Hang 0.7632（仅证明流程可执行，不作为结果） |

对照：V5 的单平面实现下，962 个训练图只有 533 个不同图像，41 个跨标签相同组覆盖 135 个样本。

## 5. 文件与运行顺序

代码文件：

- `scripts/jung_hang_models_v6.py`：Jung 单周期窗口、THD+N 注噪与限幅；Hang 标量归一化双平面轨迹。
- `scripts/compare_jung_hang_v6.py`：模型输入构建、模型入口、图像退化审计、前向自检。
- `scripts/jung_hang_optim_v6.py`：AdaMod（与 V5 逐行一致）。
- `scripts/jung_hang_cycle_v6.py`：preflight / train / evaluate / verify / summarize。
- `configs/jung_hang_v6_v1.json`：冻结配置（种子 42/44，Jung 30 epochs，Hang 50 epochs）。
- `tests/test_jung_hang_models_v6.py`：CPU 回归测试。

服务器依序执行（`REFERENCE` 必须是含 `strong_root` / `extreme_root` 的 Source-Cycle 基准部署目录）：

```bash
python scripts/jung_hang_cycle_v6.py preflight \
  --reference-root "$REFERENCE" --source-root "$SOURCE" \
  --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT"

# 四组训练，各自独立输出目录；服务器上可四进程并发
for method in jung hang; do
  for seed in 42 44; do
    python scripts/jung_hang_cycle_v6.py train --method "$method" --seed "$seed" \
      --reference-root "$REFERENCE" --source-root "$SOURCE" \
      --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT" &
  done
done
wait

for stage in evaluate verify summarize; do
  python scripts/jung_hang_cycle_v6.py "$stage" \
    --reference-root "$REFERENCE" --source-root "$SOURCE" \
    --additional-root "$ADDITIONAL" --previous-output "$PREVIOUS" --output "$OUTPUT"
done
```

## 6. 尚未验证

- 训练与七档 Dev 评价未在 GPU 服务器执行，任何诊断指标都尚未产生。
- 相序 `[0,72,144,216,288]` 未用数据列物理含义核验。
- 本地没有 pytest，单元测试是通过函数级替身运行的；服务器上应使用 `python -m pytest tests/test_jung_hang_models_v6.py` 正式执行。
