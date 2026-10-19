# Jung Modified CNN与Hang WCNN V4：默认双模型服务器实验

## Material Passport

- 模式：run；日期：2026-10-19；状态：训练、源选模、七档Dev测试和回放完成。
- 分支：`research/jung-hang-paper-fidelity-v4`；代码提交：`a0c78d8bf253082efd39b7b93428322dde5ccb67`。
- 仓库模型代码与默认配置逐字节保留；本轮未修改算法、网络及超参数。
- 两模型是论文机制指导下的五相任务适配，不是作者代码逐参数复现。

## 数据与训练

- 沿用Source-Cycle V2：63条仿真记录、962训练周期、221验证周期、78隔离周期。
- 沿用42/44种子，各50轮；Adam lr=0.001、weight_decay=0、batch32，每轮31次更新。
- 默认Jung输入为单周期五相128点，共享幅值归一化；未运行双周期配置。
- 默认Hang为两个40×40 VSD轨迹通道，不去相均值，共享窗口尺度，保留径向幅值信息。
- 两模型均使用0.5 CE(Clean)+0.5 CE(Noisy)配对训练；整记录逐相AWGN为10/15/20/25/30/35/40 dB，每轮重抽。
- 验证Clean及0/5/10/20/30 dB；使用V4普通AWGN及HUST-V4:awgn随机数命名空间。
- 原始训练噪声和采样顺序在两模型间的100个种子/轮次全部相同。
- 选模沿用源域Clean F1距峰值≤1个百分点，随后依次按五噪声平均F1、Clean F1、最小五噪声平均CE、最早轮次排序。
- 固定实测88个Dev编号，每档2474周期；5/4/0/−4/−5/−8/−10 dB均复用已有数据。
- 目标Dev不用于参数拟合或权重选择，347条Final-Holdout未评估。
- 9份受控实验预设只生成配置，未运行；本轮仅默认双模型。
- 42/44是用户查看六种子结果后指定的展示种子，本轮未增加种子。
- V2/V3/V4存在输入和训练/验证噪声变化；版本对照不是单因素消融。

## 源域输入预检

- 编译检查及26项CPU测试通过，两模型真实源样本CUDA梯度有限。
- Hang训练/干净验证均无全零图；未发现跨标签完全相同的训练图像哈希。
- [六类轨迹检查图](SOURCE_HANG_TRAJECTORIES.png)：六列依次为类别0–5；四行依次为样本平面1/2、各类均值平面1/2。
- 每类示例固定取清单中的首个训练周期，编号/边界及图像统计见[输入检查记录](SOURCE_HANG_INPUT_AUDIT.json)。
- 辅助绘图最初因环境缺少Matplotlib停止；改用NumPy及标准库生成PNG，未增加训练依赖。模型训练尚未开始时已完成修正。

## 参数量与源验证选模

| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五噪声源F1（%） |
|---|---:|---:|---:|---:|---:|
| Jung Modified CNN | 42 | 7110 | 47 | 100.00 | 97.00 |
| Jung Modified CNN | 44 | 7110 | 46 | 100.00 | 96.81 |
| Hang WCNN | 42 | 63330 | 2 | 100.00 | 98.64 |
| Hang WCNN | 44 | 63330 | 3 | 100.00 | 93.67 |

## 种子42：Macro-F1（%）

| 版本/模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 Jung Modified CNN | 87.15 | 83.97 | 65.65 | 38.33 | 33.15 | 20.88 | 16.86 |
| V3 Jung Modified CNN | 64.63 | 62.63 | 47.65 | 32.31 | 29.72 | 23.73 | 21.52 |
| V4 Jung Modified CNN | 87.70 | 84.62 | 65.38 | 41.92 | 37.48 | 28.80 | 25.67 |
| V2 Hang WCNN | 49.41 | 45.82 | 21.07 | 10.45 | 10.22 | 9.95 | 9.79 |
| V3 Hang WCNN | 7.16 | 6.86 | 6.95 | 6.41 | 5.99 | 5.24 | 5.22 |
| V4 Hang WCNN | 58.43 | 58.27 | 43.93 | 23.07 | 19.48 | 12.01 | 9.03 |

## 种子44：Macro-F1（%）

| 版本/模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 Jung Modified CNN | 88.43 | 85.27 | 66.22 | 41.67 | 35.84 | 20.42 | 15.00 |
| V3 Jung Modified CNN | 67.81 | 65.98 | 56.05 | 41.90 | 37.93 | 29.93 | 27.13 |
| V4 Jung Modified CNN | 80.05 | 79.16 | 65.55 | 44.63 | 40.02 | 31.29 | 27.59 |
| V2 Hang WCNN | 28.66 | 28.47 | 23.39 | 14.39 | 12.38 | 8.08 | 6.65 |
| V3 Hang WCNN | 13.00 | 11.67 | 8.92 | 6.50 | 5.45 | 4.86 | 4.51 |
| V4 Hang WCNN | 38.99 | 37.96 | 33.75 | 22.95 | 20.54 | 17.18 | 15.54 |

## 核验、环境及时间

- 28组目标预测重新加载best后完全一致，24组源验证F1/CE和选模记录回放一致。
- sklearn独立复算Macro-F1、Accuracy、六类F1/Recall和混淆矩阵；推理不改变模型权重和BN。
- 200轮、6200次更新全部完成；1473个历史文件指纹不变。8份新best/last留在服务器，仅发布SHA。
- 实际源记录0/5/10/20/40 dB加噪定标检查最大误差：3.55e-14 dB。
- GPU：NVIDIA GeForce RTX 5070 Ti；PyTorch `2.14.1+cu130`；复用解释器`/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`。
- 四进程训练墙钟99.553秒；测试33.508秒；核验40.338秒。
- 训练/测试/核验/汇总总墙钟174.276秒，不含预检、传输和文档整理。

## 文件

- [V4逐种子28行指标](results/reports/PER_SEED_RESULTS.csv)；[逐类168行指标](results/reports/PER_CLASS_PER_SEED.csv)。
- [V2/V3/V4逐种子对照](results/reports/V2_V3_V4_PER_SEED_COMPARISON.csv)；[引用来源指纹](results/reports/REFERENCE_PROVENANCE.json)。
- [十二旧模型与V4两模型196行合表](results/reports/ALL_14_MODELS_PER_SEED.csv)；[均值和样本标准差](results/reports/ALL_14_MODELS_SUMMARY.csv)。
- [200轮训练历史](results/reports/TRAINING_HISTORY.csv)；[完整训练/选模记录](results/models/)。
- [28份逐周期预测](results/evaluations/)；[24组源验证回放](results/reports/SOURCE_VALIDATION_REPLAY.csv)；[核验状态](results/VERIFICATION.json)。
- [原代码与配置快照](code/)；[日志](logs/)；[部署清单](DEPLOYMENT_MANIFEST.json)；[导出核验](EXPORT_VERIFICATION.json)。
- 本归档不包含原始MAT、模型权重、环境及高维NPZ。
