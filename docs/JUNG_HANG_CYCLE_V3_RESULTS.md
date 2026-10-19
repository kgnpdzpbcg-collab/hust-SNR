# Jung Modified CNN与Hang WCNN V3：服务器实验记录

## 版本与数据

- 执行日期：2026-10-19；代码分支：`research/jung-hang-faithful-adaptation-v1`；代码提交：`853b1f66a3dca6cfb480391ce0ad949024fffb06`。
- 两模型为论文机制指导下的五相任务适配，不是作者公开代码的逐参数复现。
- Source-Cycle V2：63条仿真记录，962训练周期、221验证周期、78隔离周期，全部沿用既有清单。
- 固定88个实测Dev编号，每档2474周期；测试5、4、0、−4、−5、−8、−10 dB，复用已有文件。
- 目标Dev不用于训练或选模；未评估347条Final-Holdout。

## 冻结训练及选模

- 两模型均运行42、44种子，各50轮；Adam，lr=0.001，weight_decay=0，batch32，每轮31次更新。
- Jung：每相中点去偏置、五相共享尺度，输入5×128；Clean与门控带噪样本CE各占0.5。
- Jung训练噪声：10/15/20/25/30/35/40 dB逐记录逐相AWGN，持续低电流区限制加噪，再标定整记录每相SNR。
- Hang：基波VSD平面、逐时刻模长归一化、单通道40×40图；五个卷积阶段，前两阶段后固定Haar LL池化。
- Hang仅使用Clean源样本的CE训练；未使用噪声训练视图。两模型采样顺序及更新次数相同。
- 验证：Clean与固定0/5/10/20/30 dB；V3验证也使用静默区门控噪声，随机数命名空间为HUST-V3。
- 选模：Clean F1距峰值≤1个百分点；依次按五噪声平均F1、Clean F1、最小五噪声平均CE、最早轮次排序。
- V2/V3存在输入、训练视图和源验证噪声变化；版本间差值不构成单因素消融。
- 42、44种子沿用用户此前在查看六种子结果后的指定；本轮未增加种子。

## 入口修复

- 原提交noise_records引用了setup局部导入的paper，会触发NameError；补入一行显式导入。
- 未修改模型、归一化、噪声公式或冻结超参数；修复差异见[ENTRY_FIX.patch](../records/jung_hang_cycle_v3/server_853b1f6_20261019/ENTRY_FIX.patch)。
- 新增训练/验证噪声入口回归测试；原模型8项、V3模型8项、入口回归2项共18项通过。

## 参数与源域选模

| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五噪声源F1（%） |
|---|---:|---:|---:|---:|---:|
| Jung Modified CNN | 42 | 7110 | 49 | 100.00 | 97.75 |
| Jung Modified CNN | 44 | 7110 | 48 | 100.00 | 97.71 |
| Hang WCNN | 42 | 63294 | 49 | 54.54 | 12.94 |
| Hang WCNN | 44 | 63294 | 46 | 50.86 | 13.60 |

## 种子42：Macro-F1（%）

| 版本/模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 Jung Modified CNN | 87.15 | 83.97 | 65.65 | 38.33 | 33.15 | 20.88 | 16.86 |
| V3 Jung Modified CNN | 64.63 | 62.63 | 47.65 | 32.31 | 29.72 | 23.73 | 21.52 |
| V2 Hang WCNN | 49.41 | 45.82 | 21.07 | 10.45 | 10.22 | 9.95 | 9.79 |
| V3 Hang WCNN | 7.16 | 6.86 | 6.95 | 6.41 | 5.99 | 5.24 | 5.22 |

## 种子44：Macro-F1（%）

| 版本/模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 Jung Modified CNN | 88.43 | 85.27 | 66.22 | 41.67 | 35.84 | 20.42 | 15.00 |
| V3 Jung Modified CNN | 67.81 | 65.98 | 56.05 | 41.90 | 37.93 | 29.93 | 27.13 |
| V2 Hang WCNN | 28.66 | 28.47 | 23.39 | 14.39 | 12.38 | 8.08 | 6.65 |
| V3 Hang WCNN | 13.00 | 11.67 | 8.92 | 6.50 | 5.45 | 4.86 | 4.51 |

## 核验与环境

- 两模型CUDA梯度有限；28组目标预测回放完全一致，24组源验证F1及CE回放一致。
- sklearn独立复算Macro-F1、Accuracy、各类F1/Recall及混淆矩阵；推理未改变权重和BN。
- 200轮训练、6200次更新；100个种子/轮次的模型间采样顺序一致。
- 1359个历史输入、权重和结果文件SHA未改变；8份新best/last权重留在服务器，发布SHA清单。
- 实际源记录加噪后的逐相整记录SNR最大绝对误差：4.97e-14 dB。
- GPU：NVIDIA GeForce RTX 5070 Ti；PyTorch `2.14.1+cu130`；复用解释器`/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`。
- 四进程训练墙钟：22.519秒；测试：20.100秒；核验：23.964秒。
- 完整训练/测试/回放/汇总流程：67.410秒，不含部署、预检、导出和文档整理。

## 文件

- [28行V3逐种子指标](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/PER_SEED_RESULTS.csv)；[168行逐类指标](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/PER_CLASS_PER_SEED.csv)。
- [V2/V3逐种子比较](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/V2_V3_PER_SEED_COMPARISON.csv)；[引用文件指纹](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/REFERENCE_PROVENANCE.json)。
- [十二个旧模型与V3两模型合表](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/ALL_14_MODELS_PER_SEED.csv)；[均值及样本标准差](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/ALL_14_MODELS_SUMMARY.csv)。
- [完整训练历史及选模](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/models/)；[训练历史CSV](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/TRAINING_HISTORY.csv)。
- [28份逐周期真值/预测及指标](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/evaluations/)；[源验证回放](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/reports/SOURCE_VALIDATION_REPLAY.csv)；[核验状态](../records/jung_hang_cycle_v3/server_853b1f6_20261019/results/VERIFICATION.json)。
- [代码/配置/测试快照](../records/jung_hang_cycle_v3/server_853b1f6_20261019/code/)；[训练测试日志](../records/jung_hang_cycle_v3/server_853b1f6_20261019/logs/)；[部署清单](../records/jung_hang_cycle_v3/server_853b1f6_20261019/DEPLOYMENT_MANIFEST.json)；[导出核验](../records/jung_hang_cycle_v3/server_853b1f6_20261019/EXPORT_VERIFICATION.json)。
- 原始MAT、模型权重、环境与高维NPZ不包含在发布内容中。

## 运行与归档

- [V3运行入口](../scripts/jung_hang_cycle_v3.py)；[冻结配置](../configs/jung_hang_cycle_v3_v1.json)；[入口回归测试](../tests/test_jung_hang_cycle_v3_noise.py)。
- [完整归档](../records/jung_hang_cycle_v3/server_853b1f6_20261019/SERVER_RESULTS.md)；[逐文件发布指纹](../records/jung_hang_cycle_v3/server_853b1f6_20261019/PUBLICATION_MANIFEST.json)。
- [上一版V2结果](JUNG_HANG_CYCLE_V2_RESULTS.md)；V2的代码、权重和记录均未覆盖。

从V3分支运行`jung_hang_cycle_v3.py`，复用服务器既有uv/PyTorch解释器。四个训练组分别指定`--method jung/hang`、`--seed 42/44`，随后依次执行`evaluate`、`verify`、`summarize`。所有阶段使用同一独立的新输出目录，实际命令见归档`PIPELINE_STATUS.json`。
