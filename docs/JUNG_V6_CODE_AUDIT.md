# Jung 论文复现：V6 代码核查与修订记录

## 复现定位

- 原论文：Jung et al., *Robust Open-Switch Fault Diagnosis of Three-Level NPC Inverters Based on Data Augmentation With White Noise Injection*, IEEE TPEL 40(2), 2025. DOI: 10.1109/TPEL.2024.3488094。
- 第一手依据：PDF 第 3–6 页，主要是 Section III-A/III-B（双周期窗口、归一化、CNN）、Section IV-A/B（THD+N 与故障电流噪声约束）、Section V-A（四档独立增强、AdaMod/30 epoch）。
- 这是**将原论文机制适配到 HUST 五相六类任务**的复现，并非作者源码或同拓扑精确重现。未经实际 HUST 源域审计和 GPU 训练，不能视为已完成复现。

## 逐项判定与 V6 已落实的修改

| 模块 | V5/历史偏差 | V6 措施 | 证据边界 |
|---|---|---|---|
| 两周期滑窗 | V4 正式单周期，V5 已修正两周期与索引 | 延续 V5 连续两电周期/单周期滑移及分组隔离 | 核心采样窗口符合原文 Fig. 7 |
| 归一化 | `jung_window` 对各相减 `(max+min)/2`，会把故障零电流平台移离物理零点 | 只除以**五相共享的正最大绝对值**，不平移，保持零点、符号和相间幅值比例 | 原论文仅明确缩放到 [-1,1] 并保留 polarity，没有公开准确 scale 算法；共享尺度是明示的适配 |
| THD | 在 0.10–0.45 s 非整数周期截窗直接读取单个 FFT bin，纯正弦也可能得到假谐波 | 积分电角速度确定整周期，整周期相位采样后读取整数谐波 FFT bin | 属于修复数值估计错误；原论文定义了 THD，但未公开其具体数值估计代码 |
| 训练噪声 | 在 3%、5%、7%、10% THD+N 间按轮/记录混合 | 单次运行**固定 3%**，保留可创建其他独立配置的入口 | 原文分别构建不同 THD+N 训练集，3% 是论文实验表现突出的设置 |
| 噪声实例 | 每个训练轮次重新产生随机噪声 | 每个种子只生成一份加噪训练副本，后续训练重用 | 接近原文固定倍增增强数据集；不声称与原作者 RNG 相同 |
| 故障阻断 | 对所有 OUS/OLS/OP 同样使用 RMS 低电流门限 | 仅将完全断相的 OP(A) 和复合 OP(C) 作为五相任务中明确阻断的适配对象 | 非 3L-NPC `Sj2/Sj3` 原开关限幅规则；属于五相工程适配 |
| 选模 | 以 Clean+0/5/10/20/30 dB 源验证的噪声平均 F1 选权重 | Jung 仅按**源域 Clean Macro-F1→Clean CE→较早 epoch**选权重 | 原论文未公布同一 checkpoint 排序公式；该规则为源域可复现近似 |
| 优化器 | V4 Adam | 延续 V5 AdaMod | 与 Jung 论文一致；β3 等参数细节仍需作者代码确定 |
| 训练轮数 | V4 50 | 延续 V5 30 | 与论文一致 |
| 网络结构 | V5 一层空洞卷积、两层卷积残差、GAP | 暂保留主干，不为匹配分数而随意加层 | 原文在三相 13 分类任务报告 7213 参数；现有实现还存在小差异 |
| 采样点 | V5 的每双周期统一插值至 256 点 | **尚未修正**；需要为原始 10kHz 变长序列设计可信的 batching/验证协议 | 原文采样 10kHz，输入点数随频率变化；这是后续要处理的已知复现缺口 |

### 为什么必须保留“尚未修正”

- 原论文描述的原始 10 kHz 采样率与变长输入，不能简单通过改变 `points=256` 为另一固定点数恢复。
- 要真正按原文输入，需要重构原采样时刻截窗，解决不同长度样本的 batch、BatchNorm、GAP 和源域/目标域评价一致性问题。盲目零填充会改变卷积、BN 与 GAP 的统计量，不能冒充原方法。
- 原论文 3L-NPC 开关级限幅和故障触发时刻，在五相 IM-AA 六分类任务中没有严格同构。无需虚构这些数据特性。

## 已提交的代码与测试

- `scripts/jung_hang_models_v5.py`：保留电流零点/极性的归一化；电角度整周期 THD；五相断相门控
- `scripts/jung_hang_cycle_v5.py`：单 THD+N 固定增强副本；Jung 独立源域选模
- `configs/jung_only_v6.json`：**只含 Jung**、seeds 42/44，3% THD+N
- `tests/test_jung_hang_models_v5.py`：新增非整数周期纯正弦 THD、物理零电流极性、故障相门控、选模测试
- GitHub Actions CPU 单元测试：**19 passed**。仅证明合成数据上的数值与接口约束，不是 HUST 真实数据已可用或诊断性能已提高。

## 真实 HUST 五相仿真源域（63 条 MAT）的数值核查

2026-10-21 已直接从用户上传的 `Five-phase-IM-AA-simulation-domain-data-set.rar` 读取全部 **63** 条 `I_We` (10000×6) 记录，源信号为前五相电流，第 6 列电角速度，采样率 10000 Hz。依据相同的 0.10–0.45 s 健康参考段，对照 V5 非整数周期单 FFT bin 估计与 V6 整电气周期一致采样 FFT 估计。

| 统计项 | V5 旧算法 | V6 新算法 |
|---|---:|---:|
| 记录数 | 63 | 63 |
| 被检查的相电流 | 315 | 315 |
| 记录内五相平均 THD 的跨记录中位数 | 2.1086% | 0.0285% |
| 估计 THD<3%、因而能产生非零 3% THD+N 噪声的相数 | **200/315** | **315/315** |
| 五相全部可增加 3% THD+N 噪声的记录数 | **33/63** | **63/63** |

结论：V5 的旧估计方法在源域真实数据上确有频谱泄漏问题，导致 **115/315** 相在目标 THD+N=3% 下被错误禁用增强。V6 从连续电角速度积分得到完整电周期后计算谐波幅值，已在这 63 条源记录上消除该问题。报告保存在本次会话文件 `real_source_thd.csv`；GitHub 仅归档审计结论，不上传原始 HUST 数据。此结果不代表 V6 已完成源域模型训练。

## 服务器真实数据检查步骤（尚未执行）

与 V5 相同的服务器数据目录变量：`REFERENCE`、`SOURCE`、`ADDITIONAL`、`PREVIOUS`、`OUTPUT`。使用**新目录**避免覆盖 V4/V5：

```bash
git fetch origin
git switch research/jung-paper-fidelity-audit-v6

COMMON="--config configs/jung_only_v6.json --reference-root $REFERENCE --source-root $SOURCE --additional-root $ADDITIONAL --previous-output $PREVIOUS --output $OUTPUT"
python scripts/jung_hang_cycle_v5.py preflight $COMMON
```

**首先查看** `$OUTPUT/PREFLIGHT_PAPER_AUDIT.json`：
1. `thdn_nonzero_injection_fraction_by_target_percent["3"]` 接近 1 才表明大部分记录确有注噪；若小于 1，逐条检查原生 THD 与电角速度校准。
2. `thdn_failed_records` 应为空，检查原始时间、速度、数据记录。
3. 检查多个随机源样本在归一化前后零电流平台的符号、相间峰值比例，以及实际 3% THD+N 噪声标准差。
4. 检查**Source Clean 与 target Clean** 预测变化，可先确定跨域失配，再看额外噪声退化。
5. 特别核查输入 256 点重采样是否造成故障波形形状变化；这里还没有原文级的变长输入实现。

检查无误后再运行：

```bash
for seed in 42 44; do
  python scripts/jung_hang_cycle_v5.py train --method jung --seed "$seed" $COMMON
done
python scripts/jung_hang_cycle_v5.py evaluate $COMMON
python scripts/jung_hang_cycle_v5.py verify $COMMON
```

评价只用历史约定 seed 42/44，论文的正式单 seed 展示仍使用 seed 44。**不能凭 target-Dev 性能挑选训练配置**，也不能在尚未审计时宣布 Jung 已经高保真复现。
