# B1 五项消融实验结果（种子42与44分列）

执行日期：2026-10-16。运行分支：`research/b1-paper-ablations-v1`；代码提交：`d9947f0a037c2938480b15bb1423c22e8d79a273`。

服务器使用既有uv/PyTorch环境，五个消融各完成两个种子的训练。Full B1引用先前统一基准的对应最佳权重，并重新回放核验，不重新训练。

## 数据及协议

- Source-Cycle V2：63个仿真文件，962个Train周期、221个Val周期，另隔离78个Gap周期；五相×4096点。
- Train配对Clean与每轮重新生成的10/15/20/25/30/35/40 dB噪声；Val固定Clean及0/5/10/20/30 dB。
- Adam lr=0.001、weight_decay=0、batch32、50轮联合训练；A2–A5额外10轮恢复器预训练，A1为0轮。尾部小批次处理保持原协议。
- 按源验证F1及CE选best；不以实测Dev选epoch。固定88个实测Dev编号，每条件2474个周期，评估5/0/−5/−10 dB。
- Final-Holdout未评估；42、44为用户在查看既有六种子结果后指定的展示种子。
- 下表A1–A5为本轮消融编号；A1不同于历史小波A1及单视图resnet_lp4。

## 种子42

Macro-F1（%），按保存的逐周期预测计算。

| 模型 | 5 dB | 0 dB | −5 dB | −10 dB |
|---|---:|---:|---:|---:|
| Full B1（既有最佳权重） | 97.93 | 97.08 | 92.18 | 72.32 |
| A1 固定LP4＋双视图 | 87.04 | 86.16 | 78.23 | 59.46 |
| A2 固定源训练噪声代理 | 93.69 | 92.15 | 82.30 | 65.53 |
| A3 屏蔽两组跨相描述 | 96.40 | 95.62 | 92.28 | 75.80 |
| A4 仅MD视图 | 99.79 | 97.37 | 85.82 | 59.96 |
| A5 去掉相间强度损失 | 99.11 | 95.93 | 85.19 | 61.97 |

## 种子44

Macro-F1（%），按保存的逐周期预测计算。

| 模型 | 5 dB | 0 dB | −5 dB | −10 dB |
|---|---:|---:|---:|---:|
| Full B1（既有最佳权重） | 99.70 | 97.84 | 91.42 | 71.45 |
| A1 固定LP4＋双视图 | 94.54 | 88.26 | 78.71 | 60.52 |
| A2 固定源训练噪声代理 | 95.99 | 92.04 | 84.00 | 67.81 |
| A3 屏蔽两组跨相描述 | 87.66 | 85.66 | 80.88 | 69.58 |
| A4 仅MD视图 | 89.62 | 87.51 | 77.59 | 54.22 |
| A5 去掉相间强度损失 | 94.79 | 91.80 | 88.48 | 73.94 |

## 核验及完整文件

227项代码测试和五项真实源样本CUDA梯度检查通过。60组最佳源验证回放、40组消融与8组既有Full B1预测回放一致；48组指标以sklearn独立复算。512个输入文件、84份历史权重的指纹未变。没有第二次完整训练复现。

- [服务器完整记录：配置、均值/样本SD、逐种子结果、运行耗时](../records/b1_paper_ablations/server_d9947f0_20261016/SERVER_RESULTS.md)
- [48行逐种子指标](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/WITH_FULL_B1_PER_SEED.csv)；[24行均值与样本SD](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/WITH_FULL_B1_SUMMARY.csv)
- [288行逐类F1/Recall](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/WITH_FULL_B1_PER_CLASS.csv)；[40行同种子配对差值](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/PAIRED_FULL_MINUS_ABLATION.csv)
- [500轮联合训练历史](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/TRAINING_HISTORY.csv)；[80轮恢复预训练历史](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/PRETRAIN_HISTORY.csv)
- [最佳epoch及源验证饱和统计](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/SOURCE_SELECTED_CHECKPOINTS.csv)；[源验证回放](../records/b1_paper_ablations/server_d9947f0_20261016/results/reports/SOURCE_VALIDATION_REPLAY.csv)
- [40份消融逐周期预测](../records/b1_paper_ablations/server_d9947f0_20261016/results/evaluations/)；[既有Full B1原字节引用](../records/b1_paper_ablations/server_d9947f0_20261016/full_b1_reference/)
- [运行代码/配置/测试快照](../records/b1_paper_ablations/server_d9947f0_20261016/code/)；[原始运行日志](../records/b1_paper_ablations/server_d9947f0_20261016/logs/)
- [服务器核验](../records/b1_paper_ablations/server_d9947f0_20261016/VERIFICATION.json)；[本地接收核验](../records/b1_paper_ablations/server_d9947f0_20261016/LOCAL_VERIFICATION.json)；[文件SHA256清单](../records/b1_paper_ablations/server_d9947f0_20261016/PUBLICATION_MANIFEST.json)
- 原始MAT、模型权重和环境保留在服务器；仓库记录权重SHA256及服务器路径。
- 归档代码/文档快照保留运行提交时的原字节；其中当时的待运行状态不代表本次结果状态。
