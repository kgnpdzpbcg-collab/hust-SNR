# Jung/Hang V5：服务器复现记录

## Material Passport

- 模式：run；执行日期：2026-10-21；状态：源训练、源选模、七档实测Dev测试及回放完成。
- 分支：`research/jung-hang-paper-fidelity-v5`；代码提交：`164f1c78778b30da7cc18cb71b38e154f56b750d`。
- 原模型代码、优化器和冻结配置逐字节保留；未根据目标结果改参数或重跑。
- 这是原论文机制导向的五相任务适配，不是作者官方代码的逐参数复现。

## 数据、输入与训练

- Source-Cycle V2清单仍为63条记录、962 Train、221 Val、78 Gap；源划分指纹不变。
- Jung按分组内正向连续两周期生成5×256输入，实际Train=884、Val=143；每个分组末端缺下一周期的锚点丢弃。
- Hang单周期电流经基波VSD和逐时刻矢量模长归一化，栅格为1×40×40；Train=962、Val=221。
- Hang默认quiet_ratio=0，模长≤1e-12的点跳过；未添加启发式低电流比例阈值。
- 种子42、44；Jung使用AdaMod，30轮、batch32；Hang使用Adam，50轮、batch32。两者lr=0.001、weight_decay=0。
- Jung训练使用THD+N=3/5/7/10%，健康参考段估计原生THD，故障相持续低电流区限制新增噪声。
- Hang训练使用15/20 dB逐记录逐相AWGN；两模型均为0.5 CE(Clean)+0.5 CE(Noisy)。
- 两模型均用Clean和0/5/10/20/30 dB源验证；验证噪声使用各自HUST-V5方法命名空间，非同一噪声实现。
- 源选模：Clean F1距峰值≤1个百分点，随后依次按五档平均F1、Clean F1、最小五档平均CE、最早轮次排序。
- Jung每轮28次、Hang每轮31次更新；四组共160轮、4780次更新，没有提前停止。
- 本轮未训练或评价PMSM；未读取347条Final-Holdout波形。
- 42/44种子沿用用户在查看历史六种子结果后的指定；本轮没有增加种子。

## 源预检记录

- 37项CPU测试通过，两模型真实源样本CUDA梯度有限；源输入没有全零样本。
- 63条源健康记录相位拟合最大偏差为0.01107度，支持配置中的通道顺序。此为波形证据，不是硬件接线独立认证；原审计physical标志保持false。
- 原生THD估计中位数2.1050%，最大8.1120%；来自0.10–0.45秒健康参考段FFT。
- 目标THD+N为3/5/7/10%时，315个记录×通道的非零注噪比例依次为63.49%/90.48%/93.65%/100%；估计原生THD超过目标时按代码不加噪。
- 这些比例依赖当前FFT估计方法，不是独立仪器标定的THD测量。
- Hang源训练962个样本中有533种不同图像；跨标签完全相同图像共有41组、涉及135个样本。
- [源输入检查图](../records/jung_hang_cycle_v5/server_164f1c7_20261021/SOURCE_HANG_INPUTS.png)：列依次为0–5标签，上行为每类首个训练样本，下行为类平均图。
- [原分支源预检](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/PREFLIGHT_PAPER_AUDIT.json)；[相序拟合](../records/jung_hang_cycle_v5/server_164f1c7_20261021/SOURCE_PHASE_ORDER_AUDIT.json)；[预检审阅](../records/jung_hang_cycle_v5/server_164f1c7_20261021/PREFLIGHT_REVIEW.json)；[同图跨标签统计](../records/jung_hang_cycle_v5/server_164f1c7_20261021/SOURCE_HANG_INPUT_AUDIT.json)。

## 参数与最佳权重

| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五档源F1（%） |
|---|---:|---:|---:|---:|---:|
| Jung Modified CNN V5 | 42 | 7110 | 10 | 100.00 | 90.88 |
| Jung Modified CNN V5 | 44 | 7110 | 27 | 100.00 | 79.39 |
| Hang WCNN V5 | 42 | 63294 | 48 | 54.35 | 9.50 |
| Hang WCNN V5 | 44 | 63294 | 48 | 56.46 | 9.28 |

## 共同目标评价队列

- 原固定88个Dev编号、2474个周期中，共2298个锚点具有相邻下一周期。两模型、两种子和七档全部使用同一2298个锚点。
- 真值类别数依次为[1018, 289, 279, 280, 220, 212]。Jung看连续两周期，Hang看锚点单周期。
- 七档为5/4/0/−4/−5/−8/−10 dB，复用已有固定噪声文件；实测不用于参数训练或选模。
- 历史十二模型使用2474周期。上游汇总入口附带的ALL_14文件保持原样，是不同评价队列的原始拼接，本记录没有据此作跨版本/跨模型排名。见[队列说明](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/reports/COHORT_NOTICE.json)。

## 种子42：共同2298锚点Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jung Modified CNN V5 | 58.91 | 54.71 | 39.30 | 25.57 | 22.96 | 19.34 | 18.17 |
| Hang WCNN V5 | 2.91 | 2.91 | 2.91 | 2.91 | 2.91 | 2.91 | 2.91 |

## 种子44：共同2298锚点Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jung Modified CNN V5 | 40.40 | 33.77 | 17.14 | 12.92 | 11.85 | 10.50 | 9.60 |
| Hang WCNN V5 | 2.91 | 2.91 | 2.91 | 2.91 | 2.91 | 2.91 | 2.91 |

## 预测、核验与环境

- Hang两个种子的七档预测均为：全部2298样本预测标签4。Accuracy=9.5735%，Macro-F1=2.9124%。逐周期预测完整保留。
- 28组best目标预测回放、24组best源验证F1/CE和源选模回放均通过；推理未改变权重或BN。
- sklearn独立复算宏F1、准确率、各类F1/Recall及混淆矩阵；另外从冻结元数据重新生成共同锚点并逐文件核对。
- 1608个历史数据、代码、权重和结果文件SHA保持不变；8份新权重留服务器，只发布SHA清单。
- GPU：NVIDIA GeForce RTX 5070 Ti；PyTorch `2.14.1+cu130`；复用解释器`/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`，未新建环境。
- 四进程训练墙钟78.030秒；测试21.116秒；核验24.541秒。
- 训练、测试、核验、汇总总墙钟124.661秒，不含预检、传输及整理。

## 文件

- [28行逐种子指标](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/reports/PER_SEED_RESULTS.csv)；[168行逐类指标](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/reports/PER_CLASS_PER_SEED.csv)。
- [160轮训练历史](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/reports/TRAINING_HISTORY.csv)；[完整训练及选模](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/models/)；[24组源验证回放](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/reports/SOURCE_VALIDATION_REPLAY.csv)。
- [28份逐周期真值、预测及原周期锚点](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/evaluations/)；[核验状态](../records/jung_hang_cycle_v5/server_164f1c7_20261021/results/VERIFICATION.json)。
- [代码、配置与测试快照](../records/jung_hang_cycle_v5/server_164f1c7_20261021/code/)；[日志](../records/jung_hang_cycle_v5/server_164f1c7_20261021/logs/)；[部署指纹](../records/jung_hang_cycle_v5/server_164f1c7_20261021/DEPLOYMENT_MANIFEST.json)；[导出核验](../records/jung_hang_cycle_v5/server_164f1c7_20261021/EXPORT_VERIFICATION.json)。
- 原始MAT、模型权重、环境及高维特征不包含在发布内容中。

## 运行入口与归档

- [V5入口](../scripts/jung_hang_cycle_v5.py)；[冻结配置](../configs/jung_hang_v5_v1.json)；[论文机制与适配边界](JUNG_HANG_V5_PAPER_FIDELITY_REVIEW.md)。
- [完整归档](../records/jung_hang_cycle_v5/server_164f1c7_20261021/SERVER_RESULTS.md)；[发布逐文件指纹](../records/jung_hang_cycle_v5/server_164f1c7_20261021/PUBLICATION_MANIFEST.json)。

服务器复用既有uv/PyTorch解释器。先运行并审阅`preflight`；四个`train`分别指定`--method jung/hang`、`--seed 42/44`；随后按`evaluate`、`verify`、`summarize`顺序执行。全部阶段共用同一个独立输出目录和冻结配置，实际命令见归档`PIPELINE_STATUS.json`。
