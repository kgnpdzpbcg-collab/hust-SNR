# Jung/Hang V4 — 论文对比模型复现改进与受控验证

状态：**2026-10-19已完成默认双模型42/44种子的四组50轮训练、28组七档Target-Dev评价及预测回放。** 见[V4服务器客观结果](JUNG_HANG_CYCLE_V4_RESULTS.md)。九个消融预设仅生成配置，未运行。历史V2/V3文件、结果、权重未覆盖。

## 1. 为什么建立 V4

- Jung (IEEE TPEL 2025): 源域能学习，但单周期 128 点的五相适配不等于论文的两周期滑动输入。检验周期数、归一化和 SNR 噪声增强，保留 dilated Conv1D + residual + GAP 分类主干。
- Hang (IEEE TIE 2023): V3 源域 Clean Macro-F1 约 51–55%，不足以支持可信跨域对比。先验证 VSD 电流轨迹是否保留故障差异；再检验 WCNN+DWT 与噪声训练策略。原文采用三相 Clarke 轨迹；此处是**五相 VSD 任务适配，不是精确作者代码复现**。

## 2. 修改的实现与默认设置

- `scripts/jung_hang_models_v4.py`: Jung 输入与 Hang 轨迹适配器；复用 V3 的 CNN、固定 Haar LL、五相 VSD 定义；增加可选按记录定标 SNR-AWGN。
- `scripts/compare_jung_hang_v4.py`: 输入构造、双周期边界检查、网络工厂、CPU smoke。
- `scripts/jung_hang_cycle_v4.py`: 统一协议下 `preflight/train/evaluate/verify/summarize`，支持单模型或双模型配置。
- `configs/jung_hang_cycle_v4_v1.json`: 默认两模型；Jung 单周期+共享幅值+AWGN；Hang 双 VSD 轨迹+不去均值+窗口共享幅值+配对 AWGN。
- `scripts/jung_hang_v4_presets.py`: 自动生成**9 个独立实验配置**，不会运行实验。
- `tests/test_jung_hang_models_v4.py`: 模型形状、梯度、双周期边界、单/双 VSD、去均值、噪声定标等 CPU 检查。
- `.github/workflows/jung-hang-v4-tests.yml`: 仅编译与合成数据 CPU 测试，不访问项目真实原始数据。

训练样本、划分及训练/验证 SNR 继续沿用 Source-Cycle V2（962 train、221 val、78 gap；seed 42/44）。模型选择仍只使用源域验证。目标域七档仅用于测试，不作为训练或 epoch 选择输入。

### Jung 关键实现边界

**双周期输入**仅在当前数据划分内部、同一 `record_id`、同一 `label`、紧邻周期中拼接；不允许跨 Train/Val gap、不同故障标签或硬拼接不同记录。对缺乏相邻周期的窗口直接报错。因此必须先运行 `preflight` 检查 J0 双周期配置能否覆盖现有划分；若不能，不要默默退化成单周期或重复同一周期，而应明确重新设计样本协议。

### Hang 关键实现边界

- 不去均值 (`hang_center=none`) 可保留直流及相间不对称信息；并不保证是最优。
- `planes=fundamental` 与 `planes=dual` 均使用两通道 40×40 WCNN，前者第二通道固定为零，确保网络容量不因通道数而改变。
- 共享幅值归一化保留轨迹径向信息，instantaneous 逐时刻模长归一化更接近原文式(5)/(6)。两者都必须做故障类别轨迹可视化与源域结果核查。
- 对 instant 模式中低模长点的阈值采用窗口峰值的 3%；无效点不会穿线连接。该阈值是明确的五相工程适配，不能冒充原论文逐参数配置。
- V4 的 Hang 默认 `paired` 训练和 V3 的 `clean-only` 训练分别保留为受控选项；不把其结果差异归因于网络结构。

## 3. 推荐执行顺序

1. 本地/CI 检查：`python -m py_compile scripts/jung_hang_models_v4.py scripts/compare_jung_hang_v4.py scripts/jung_hang_cycle_v4.py`；`pytest -q tests/test_jung_hang_models_v4.py`。
2. 生成配置：`python scripts/jung_hang_v4_presets.py --output outputs/jung_hang_v4_configs`。
3. 逐配置进行 `preflight`，仅加载仿真训练/验证集并确认输入形状、配对周期和 CPU smoke；**双周期失败即停**。
4. 先对 Hang 检查 6 类 VSD 轨迹与源域可分性，再用两个种子依次完成 H0/H1/H2/H3；Jung 依次 J0/J1/J2。只变更一个因素，不根据 Target-Dev 曲线随意调整参数。
5. 从源验证选择权重，运行七档 Dev 的 `evaluate`、`verify` 和 `summarize`。保留全部逐周期预测、混淆矩阵、训练历史与配置哈希；不触碰 Final-Holdout。
6. 选定最终两个高质量基线适配版本后，才更新会议论文主表。**原论文期刊质量不意味着跨任务必然表现优异。**

### V4 命令示例

下面的路径需要替换为服务器上既有 V3 实验对应的目录，并为每套配置使用不同输出路径。

```bash
python scripts/jung_hang_v4_presets.py --output outputs/jung_hang_v4_configs

python scripts/jung_hang_cycle_v4.py preflight \
  --config outputs/jung_hang_v4_configs/hang_h0_dual_no_center.json \
  --reference-root /ABS/REFERENCE_BENCHMARK \
  --source-root /ABS/SOURCE_CYCLE_V2 \
  --additional-root /ABS/ADDITIONAL_SNR \
  --previous-output /ABS/TWELVE_MODEL_OUTPUT \
  --output /ABS/NEW_HANG_H0

python scripts/jung_hang_cycle_v4.py train --method hang --seed 42 \
  --config outputs/jung_hang_v4_configs/hang_h0_dual_no_center.json \
  --reference-root /ABS/REFERENCE_BENCHMARK \
  --source-root /ABS/SOURCE_CYCLE_V2 \
  --additional-root /ABS/ADDITIONAL_SNR \
  --previous-output /ABS/TWELVE_MODEL_OUTPUT \
  --output /ABS/NEW_HANG_H0
```

再用相同配置和输出位置训练 seed 44，并执行 `evaluate`、`verify`、`summarize`。单方法的配置只需要训练该方法，其他模型不会被重新训练。不要复用其他配置下的输出目录。

## 4. 验收边界

本提交的 CPU 测试只证明逻辑一致性、张量形状、边界检查及数值正常；**不是源域诊断性能验证**。真正能否提高 Jung/Hang 性能，必须等待服务器重新训练及预测记录。J0 的双周期输入使用不同时间支持，若与单周期对照，样本覆盖及源域窗口条件必须额外审查。

参考：Jung et al., *IEEE TPEL*, 2025, DOI 10.1109/TPEL.2024.3488094；Hang et al., *IEEE TIE*, 2023, DOI 10.1109/TIE.2022.3231304。
