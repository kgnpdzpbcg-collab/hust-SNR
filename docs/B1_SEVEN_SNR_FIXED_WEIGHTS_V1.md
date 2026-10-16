# 八模型七档SNR结果：完整B1、A1–A5、Xu-Full、B0

> ResNet1D、Plain CNN1D、ResNet＋LP3、ResNet＋LP4已补齐并归档，见[十二模型七档完整结果](TWELVE_MODEL_SEVEN_SNR_V1.md)。下文保留原八模型记录。

## Material Passport

- 执行日期：2026-10-16（Asia/Shanghai）；模式：固定权重推理、预测回放、汇总。
- 核验范围：42组新增预测及70组历史预测均回放，112组指标独立复算；没有重新训练。
- 模型：完整B1、五个B1消融、Xu-Full、B0；指定种子42、44。
- 原完整B1/Xu/B0训练提交：`4728e9b82c4c181564f0247aa4b1028a35a8e8d2`。
- 原五个消融训练提交：`d9947f0a037c2938480b15bb1423c22e8d79a273`。

## 输入与执行协议

- 复用Source-Cycle V2的962 Train、221 Val，原每相4096点；本轮不训练、不选新epoch。
- 使用原16份源验证best，严格加载完整state_dict，包括BN缓冲区和A2固定噪声代理。
- 训练条件仍为原10–40 dB配对增强、Clean及0/5/10/20/30 dB源验证；没有修改原配置。
- 固定88个实测Dev编号，七档5/4/0/−4/−5/−8/−10 dB，每档2474个周期；六类数量[1106,307,297,298,237,229]。
- 已有4/−4/−8 dB的264份MAT直接复用，并重新核查作者残差缩放、SHA256、时间及转速列。
- 264份输入的整记录逐相SNR最大误差：3.863576125695545e-14 dB。
- SNR定义按原整记录逐相计算；不同档位共享同一作者10 dB残差，不是独立噪声实现。
- A1–A5、Xu-Full、B0共7方法×2种子×3档，新增42组；其余70份历史预测按原字节引用。
- A1为本轮固定LP4＋双视图消融，不是历史小波A1，也不同于单视图resnet_lp4。
- 42/44为用户在查看此前六种子结果后指定的子集；此前全部六种子记录保留。
- 本轮不读取Final-Holdout波形，不进行t-SNE或重新训练。

## 种子42：Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整B1 | 97.93 | 97.72 | 97.08 | 94.34 | 92.18 | 81.39 | 72.32 |
| A1 固定LP4＋双视图 | 87.04 | 87.24 | 86.16 | 80.03 | 78.23 | 67.10 | 59.46 |
| A2 固定噪声估计 | 93.69 | 93.87 | 92.15 | 85.01 | 82.30 | 73.25 | 65.53 |
| A3 屏蔽两组跨相描述 | 96.40 | 96.52 | 95.62 | 93.88 | 92.28 | 85.67 | 75.80 |
| A4 仅MD视图 | 99.79 | 99.54 | 97.37 | 89.30 | 85.82 | 71.53 | 59.96 |
| A5 去掉相间强度损失 | 99.11 | 98.97 | 95.93 | 87.87 | 85.19 | 72.11 | 61.97 |
| Xu-Full | 60.37 | 54.92 | 31.39 | 14.60 | 12.62 | 9.28 | 9.11 |
| B0 | 91.50 | 90.45 | 81.47 | 63.68 | 58.52 | 42.43 | 31.81 |

## 种子44：Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整B1 | 99.70 | 99.45 | 97.84 | 93.18 | 91.42 | 82.00 | 71.45 |
| A1 固定LP4＋双视图 | 94.54 | 93.77 | 88.26 | 80.96 | 78.71 | 68.33 | 60.52 |
| A2 固定噪声估计 | 95.99 | 95.48 | 92.04 | 85.98 | 84.00 | 74.65 | 67.81 |
| A3 屏蔽两组跨相描述 | 87.66 | 87.71 | 85.66 | 81.90 | 80.88 | 75.98 | 69.58 |
| A4 仅MD视图 | 89.62 | 89.78 | 87.51 | 81.48 | 77.59 | 65.48 | 54.22 |
| A5 去掉相间强度损失 | 94.79 | 94.28 | 91.80 | 89.48 | 88.48 | 82.17 | 73.94 |
| Xu-Full | 77.78 | 75.13 | 54.80 | 34.19 | 30.20 | 22.05 | 19.06 |
| B0 | 99.27 | 99.02 | 95.36 | 83.38 | 78.90 | 57.95 | 43.77 |

## 两个指定种子的均值±样本标准差（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 完整B1 | 98.81 ± 1.25 | 98.58 ± 1.22 | 97.46 ± 0.54 | 93.76 ± 0.82 | 91.80 ± 0.53 | 81.69 ± 0.43 | 71.88 ± 0.61 |
| A1 固定LP4＋双视图 | 90.79 ± 5.31 | 90.51 ± 4.62 | 87.21 ± 1.48 | 80.49 ± 0.66 | 78.47 ± 0.34 | 67.72 ± 0.87 | 59.99 ± 0.75 |
| A2 固定噪声估计 | 94.84 ± 1.62 | 94.68 ± 1.14 | 92.09 ± 0.08 | 85.49 ± 0.69 | 83.15 ± 1.20 | 73.95 ± 0.99 | 66.67 ± 1.61 |
| A3 屏蔽两组跨相描述 | 92.03 ± 6.18 | 92.12 ± 6.23 | 90.64 ± 7.05 | 87.89 ± 8.47 | 86.58 ± 8.06 | 80.82 ± 6.85 | 72.69 ± 4.40 |
| A4 仅MD视图 | 94.70 ± 7.19 | 94.66 ± 6.90 | 92.44 ± 6.97 | 85.39 ± 5.53 | 81.71 ± 5.82 | 68.50 ± 4.28 | 57.09 ± 4.06 |
| A5 去掉相间强度损失 | 96.95 ± 3.06 | 96.62 ± 3.31 | 93.86 ± 2.92 | 88.68 ± 1.13 | 86.84 ± 2.32 | 77.14 ± 7.11 | 67.95 ± 8.46 |
| Xu-Full | 69.07 ± 12.30 | 65.03 ± 14.29 | 43.09 ± 16.55 | 24.40 ± 13.85 | 21.41 ± 12.43 | 15.66 ± 9.03 | 14.09 ± 7.03 |
| B0 | 95.38 ± 5.49 | 94.74 ± 6.06 | 88.41 ± 9.83 | 73.53 ± 13.93 | 68.71 ± 14.41 | 50.19 ± 10.98 | 37.79 ± 8.46 |

## 运行与核验

- GPU：NVIDIA GeForce RTX 5070 Ti；PyTorch `2.14.1+cu130`；复用解释器`/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`。
- 16份best真实源样本前向检查通过；96组原最佳源验证F1/CE重放一致。
- 112组固定Dev逐周期预测回放一致；112组六类F1、Recall、Accuracy、混淆矩阵、健康误报和故障漏报使用sklearn独立复算。
- 70份被引用的历史预测与原文件逐字节一致，42份新增预测保留完整真值和预测。
- 原84份统一基准权重及20份消融权重的SHA核验通过；推理未改变16个模型的参数或BN。
- 1126个受保护文件在执行前后指纹一致。
- 以下为各阶段子进程墙钟时间，不含部署、传输及文档整理。

| 阶段 | 秒 | 退出码 |
|---|---:|---:|
| preflight | 4.132 | 0 |
| evaluate | 5.389 | 0 |
| verify | 11.617 | 0 |
| summarize | 0.114 | 0 |

- 四阶段合计：21.251秒。

## 结果文件

- [112行逐种子结果](../records/b1_seven_snr_fixed_weights/server_20261016/output/results/reports/PER_SEED_RESULTS.csv)；[56行均值/样本SD](../records/b1_seven_snr_fixed_weights/server_20261016/output/results/reports/SUMMARY.csv)。
- [672行六类指标](../records/b1_seven_snr_fixed_weights/server_20261016/output/results/reports/PER_CLASS_PER_SEED.csv)；[98行同种子完整B1配对差值](../records/b1_seven_snr_fixed_weights/server_20261016/output/results/reports/PAIRED_FULL_B1_MINUS_MODELS.csv)。
- [16份最佳权重身份与轮次](../records/b1_seven_snr_fixed_weights/server_20261016/output/results/reports/SELECTED_CHECKPOINTS.csv)；[96组源验证回放](../records/b1_seven_snr_fixed_weights/server_20261016/output/results/reports/SOURCE_VALIDATION_REPLAY.csv)。
- [全部112份逐周期预测](../records/b1_seven_snr_fixed_weights/server_20261016/output/results/evaluations/)；[每组结果来源](../records/b1_seven_snr_fixed_weights/server_20261016/output/RESULT_PROVENANCE.json)。
- [核验记录](../records/b1_seven_snr_fixed_weights/server_20261016/output/VERIFICATION.json)；[数据来源及264份MAT指纹](../records/b1_seven_snr_fixed_weights/server_20261016/output/DATA_REFERENCE.json)。
- [原训练元数据](../records/b1_seven_snr_fixed_weights/server_20261016/output/checkpoint_metadata/)；[本次入口与配置](../records/b1_seven_snr_fixed_weights/server_20261016/code/)；[运行日志](../records/b1_seven_snr_fixed_weights/server_20261016/logs/)。
- 原MAT、模型权重和环境留在服务器。

## 本次入口及复用方法

- [独立补测入口](../scripts/b1_seven_snr_fixed_weights.py)；[冻结配置](../configs/b1_seven_snr_fixed_weights_v1.json)。
- [服务器完整记录](../records/b1_seven_snr_fixed_weights/server_20261016/SERVER_RESULTS.md)；[逐文件发布SHA256](../records/b1_seven_snr_fixed_weights/server_20261016/PUBLICATION_MANIFEST.json)。

复用服务器现有环境及三个既有实验目录。运行下列四阶段时使用同一个全新`--output`，阶段依次为`preflight`、`evaluate`、`verify`、`summarize`；任何失败均停止，不覆盖原实验。

```bash
python scripts/b1_seven_snr_fixed_weights.py preflight \
  --baseline-root /ABS/PATH/experiments/source_cycle_benchmark/4728e9b \
  --ablation-root /ABS/PATH/experiments/b1_paper_ablations/d9947f0_20261016 \
  --additional-root /ABS/PATH/experiments/additional_snr_seed42_44/20261015_snr4_m4_m8 \
  --source-root /ABS/PATH/datasets/source_cycle_split_v2 \
  --output /ABS/PATH/NEW_FIXED_WEIGHT_OUTPUT
```
