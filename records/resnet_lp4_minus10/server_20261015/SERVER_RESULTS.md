# LP4＋ResNet固定权重−10 dB评估记录

## Material Passport

- 类型：服务器实际固定权重推理及数值核验记录；客户端日期：2026-10-15。
- 来源：本目录配置、运行日志、results/evaluations逐周期预测及results/reports原始汇总。
- 状态：3份旧权重实际推理完成；历史−5 dB预测复现及新−10 dB重复推理均通过。

## 固定数据与权重

仅评估原88个实测Dev文件中的2474个周期，类别0–5周期数为1106/307/297/298/237/229；切窗编号、边界、读取去重、周期插值、LP4、相电流归一化及CNN输入处理全部调用原训练代码快照。LP4保留DC及1–4阶谐波，模型参数241702。

- 原训练代码提交：`5c75bea67ffb1dd7730398193aba8b3498382e82`。
- 本轮入口代码提交：`cb14cd14b5dc64ae6bfc26413db570a7fbb4d636`。
- 原权重：`/home/wang/projects/会议论文/hust/experiments/architecture_suite/5c75bea/results/models/resnet_lp4/seed{42,43,44}/best.pt`。
- 新输入：`/home/wang/projects/会议论文/hust/datasets/strong_snr_minus10_v1_20261015/generated/target/SNR_-10/`。
- 本轮结果：`/home/wang/projects/会议论文/hust/experiments/resnet_lp4_minus10/20261015/results/`。
- 原环境：`/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`；复用既有PyTorch/CUDA，没有安装依赖。

原权重是在34份仿真训练、15份仿真验证上完成50轮训练后选出，训练噪声10–40 dB，选模源验证为Clean与0/5/10/20/30 dB。本轮不训练、不微调、不改变选中轮次；新生成的仿真−10 dB没有用于训练或验证。不读取实测最终保留集347份波形进行推理，也不读取仿真Source-Test做评估。

## Macro-F1与Accuracy（%，三个种子均值±样本标准差）

| 噪声条件 | Macro-F1 | Accuracy |
|---|---:|---:|
| clean | 99.42 ± 0.50 | 99.62 ± 0.33 |
| 10 | 99.00 ± 0.68 | 99.31 ± 0.39 |
| 5 | 98.10 ± 0.85 | 98.53 ± 0.35 |
| 0 | 94.44 ± 1.05 | 95.12 ± 1.90 |
| -5 | 80.43 ± 2.92 | 82.59 ± 4.13 |
| -10 | 51.81 ± 3.43 | 55.48 ± 4.35 |

Clean/10/5/0 dB来自原架构实验记录；−5 dB来自上轮固定权重记录；−10 dB为本轮新推理。−10 dB相比−5 dB的平均Macro-F1下降28.62个百分点。0/5/−5/−10 dB均为合成实测压力条件，不能称为独立采集数据。

## −10 dB逐种子记录

| 种子 | 原选中轮次 | Macro-F1（%） | Accuracy（%） | 健康误判为故障（/1106） | 故障误判为健康（/1368） |
|---|---:|---:|---:|---:|---:|
| 42 | 7 | 47.97 | 50.61 | 588 | 64 |
| 43 | 2 | 52.89 | 56.87 | 528 | 50 |
| 44 | 5 | 54.57 | 58.97 | 482 | 36 |

## −10 dB各类别（%，均值±样本标准差）

| 标签 | 原类别 | F1 | Recall |
|---|---|---:|---:|
| 0 | Healthy | 66.24 ± 4.66 | 51.84 ± 4.81 |
| 1 | OUS(A) | 23.80 ± 2.38 | 23.45 ± 7.68 |
| 2 | OLS(A) | 45.70 ± 5.95 | 53.87 ± 16.66 |
| 3 | OP(A) | 67.83 ± 10.97 | 81.21 ± 7.17 |
| 4 | OUS(A)&OUS(B) | 52.67 ± 5.75 | 56.96 ± 18.66 |
| 5 | OLS(A)&OP(C) | 54.62 ± 2.85 | 83.11 ± 26.99 |

每个种子的混淆矩阵均保存在对应`target_dev_snr-10.json`，行是真实类别，列是预测类别。上表是分别计算三个种子后取均值，不是先合并三份混淆矩阵再计算。

## 核验

- 三份原权重SHA与原训练记录一致，加载后全部参数及缓冲区逐数值不变。
- 三组−5 dB历史预测均在原2474个周期上逐项复现；三组−10 dB两次推理预测完全一致。
- 15组历史指标及3组新增指标经sklearn独立复算一致；631个受保护文件SHA不变。
- 原确定性CUDA设置不变：deterministic=True，cuDNN与matmul的TF32=False；推理批大小128。
- 499份数据生成步骤和3份权重推理分目录保存。评估脚本总耗时2.448秒，包含数据核验、−5 dB重放及−10 dB重复推理，不是训练用时。

[逐种子及六档对照](results/reports/PER_SEED_RESULTS.csv) · [逐类指标](results/reports/PER_CLASS_PER_SEED.csv) · [配对退化](results/reports/PAIRED_DEGRADATION.csv) · [独立核验](results/reports/VERIFICATION.json) · [数据生成记录](../../strong_snr_minus10/server_20261015/GENERATION_RECORD.md)
