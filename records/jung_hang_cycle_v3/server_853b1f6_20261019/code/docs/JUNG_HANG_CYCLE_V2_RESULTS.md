# Jung Modified CNN与Hang WCNN：统一协议实验结果

## Material Passport

- 执行日期：2026-10-19（Asia/Shanghai）；模式：源训练、固定权重目标Dev测试、预测回放。
- 模型代码分支：`research/noise-consistency-v1`；模型提交：`2b0ab5a1b0d4ea16ab68d3ce56b5c69a4f30218d`。
- 网络与输入适配器为仓库近似复现/任务适配，不是作者发布的逐参数原实现。
- 模型结构及`compare_jung_hang.build_inputs`未修改；新增独立统一协议入口。

## 数据与协议

- Source-Cycle V2：63条原始仿真记录，962 Train、221 Val、78 Gap，沿用原周期边界。
- 种子42、44；50轮、Adam lr=0.001、weight_decay=0、batch32，每轮31次更新；尾部2个基础样本批次处理保持一致。
- 两模型都使用Clean/Noisy配对CE，各权重0.5；整记录逐相10/15/20/25/30/35/40 dB增强，每轮重抽。
- 同种子、轮次的原始噪声SHA与采样顺序在两模型间完全相同；第1轮噪声与4728e9b基准逐字节一致。
- 验证为Clean及固定0/5/10/20/30 dB；原始验证噪声逐周期与既有基准相同。
- 选模：Clean F1距峰值≤1个百分点；五噪声平均F1、Clean F1、最小五噪声平均CE、最早epoch依次排序。
- Jung原输入为五相MD归一化后128点；Hang原输入为五相128点VSD轨迹、两个64×64通道。
- 与新分支默认训练器不同：采用当前周期划分、42/44种子、五档验证及CE平局规则，并将Hang改为配对噪声训练。论文分支原训练器仍保留。
- 该比较统一原始数据、噪声、更新次数和选模；两个方法的输入表示不同。
- 固定88个实测Dev编号、每档2474周期，测试5/4/0/−4/−5/−8/−10 dB；不重新生成目标噪声。
- 目标Dev仅在源训练和源选模结束后评估；不读取Final-Holdout波形。
- 展示种子42/44来自用户此前在查看六种子结果后的指定，不替代六种子稳定性研究。

## 参数量与最佳轮次

| 模型 | 种子 | 参数量 | 源验证选中的epoch |
|---|---:|---:|---:|
| Jung Modified CNN | 42 | 7110 | 33 |
| Jung Modified CNN | 44 | 7110 | 32 |
| Hang WCNN | 42 | 123234 | 48 |
| Hang WCNN | 44 | 123234 | 50 |

## 种子42：Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jung Modified CNN | 87.15 | 83.97 | 65.65 | 38.33 | 33.15 | 20.88 | 16.86 |
| Hang WCNN | 49.41 | 45.82 | 21.07 | 10.45 | 10.22 | 9.95 | 9.79 |

## 种子44：Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jung Modified CNN | 88.43 | 85.27 | 66.22 | 41.67 | 35.84 | 20.42 | 15.00 |
| Hang WCNN | 28.66 | 28.47 | 23.39 | 14.39 | 12.38 | 8.08 | 6.65 |

## 检查、环境与时间

- 原模型8项CPU测试通过；两模型真实源样本CUDA梯度检查通过。
- Jung参数量7110；Hang参数量123234。模型默认VSD相轴为0/72/144/216/288度。
- 健康源样本基波相对相位中位数为[0.0, -72.00437927246094, -144.00103759765625, 143.99853515625, 71.99119567871094]度；相轴未调节。
- 28组Dev预测重新加载best后完全一致；24组最佳源验证F1/CE回放一致；原空间六类指标以sklearn独立复算。
- 同种子两模型全部100个种子/epoch的噪声及采样SHA一致；200轮训练、6200次参数更新，未提前停止。
- 1304个旧输入、历史权重及预测文件的指纹未变；推理不改变模型参数或BN。
- GPU：NVIDIA GeForce RTX 5070 Ti；PyTorch `2.14.1+cu130`；解释器`/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`。
- 四进程并发训练墙钟时间：103.557秒；测试：32.105秒；核验：37.478秒。
- 训练/测试/核验/汇总流程：173.967秒，不含部署、代码检查、传输和文档整理。

## 文件

- [28行新模型逐种子指标](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/PER_SEED_RESULTS.csv)；[168行新模型逐类指标](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/PER_CLASS_PER_SEED.csv)。
- [196行十四模型合表](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/ALL_14_MODELS_PER_SEED.csv)；[98行十四模型均值/样本SD](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/ALL_14_MODELS_SUMMARY.csv)。
- [200轮训练历史](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/TRAINING_HISTORY.csv)；[完整训练/选模记录](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/models/)；[24组源验证回放](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/SOURCE_VALIDATION_REPLAY.csv)。
- [28份逐周期真值/预测](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/evaluations/)；[核验状态](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/VERIFICATION.json)；[旧十二模型引用来源](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/REFERENCE_PROVENANCE.json)。
- [运行代码、原模型、配置及测试快照](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/code/)；[训练和测试日志](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/logs/)。
- 原始MAT、模型best/last及环境留在服务器，发布权重SHA清单。

## 运行入口与原始对照记录

- [统一协议入口](../scripts/jung_hang_cycle_v2.py)；[冻结配置](../configs/jung_hang_cycle_v2_v1.json)。
- [原十二模型七档完整结果](https://github.com/kgnpdzpbcg-collab/hust-SNR/blob/research/b1-paper-ablations-v1/docs/TWELVE_MODEL_SEVEN_SNR_V1.md)。
- [本轮完整归档](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/SERVER_RESULTS.md)；[逐文件发布指纹](../records/jung_hang_cycle_v2/server_2b0ab5a_20261019/PUBLICATION_MANIFEST.json)。

从当前分支运行`jung_hang_cycle_v2.py`；服务器原环境不重建。四阶段`train`、`evaluate`、`verify`、`summarize`使用同一个新输出目录。`train`须分别指定`--method jung/hang`及`--seed 42/44`。

```bash
python scripts/jung_hang_cycle_v2.py train --method jung --seed 42 \
  --reference-root /ABS/PATH/experiments/source_cycle_benchmark/4728e9b \
  --source-root /ABS/PATH/datasets/source_cycle_split_v2 \
  --additional-root /ABS/PATH/experiments/additional_snr_seed42_44/20261015_snr4_m4_m8 \
  --previous-output /ABS/PATH/experiments/baseline_four_snr_completion/20261016/output \
  --output /ABS/PATH/NEW_JUNG_HANG_RESULTS
```
