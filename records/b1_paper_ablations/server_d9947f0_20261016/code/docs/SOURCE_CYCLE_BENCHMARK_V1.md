# 新周期划分：七模型六种子实验V1

状态：2026-10-15服务器42组训练和504组实测Dev评价已完成，保存权重的252组源验证和504组实测预测重放通过；最终Holdout未评价。完整记录见[服务器报告](../records/source_cycle_benchmark/server_4728e9b/SERVER_RESULTS.md)。

- 数据：全63文件、13工况的新周期划分，Train=962、Val=221、Gap=78。样本仍为五相完整周期，每相4096点。
- 模型：原ResNet1D、Plain CNN1D、Xu-Full、ResNet+LP3、ResNet+LP4、B0、B1；均直接导入原模型类和损失函数，没有修改模型算法。
- 种子：42、43、44、2024、2025、2026，每模型6组，共42组源训练。
- 共同配置：Adam lr=0.001、weight_decay=0、batch32、50轮；每轮Clean与10/15/20/25/30/35/40 dB新噪声配对，所有模型同种子同轮次使用相同噪声与采样顺序。
- 验证：Clean与0/5/10/20/30 dB固定噪声，仅使用仿真Val。
- 统一选模：Clean F1在峰值1个百分点内，依次最大化五噪声平均F1、Clean F1；F1并列时最小化五噪声平均CE，再选较早轮次。此规则沿用B1；历史其他模型的最早F1并列规则不用于本轮。
- 模型损失：ResNet、Plain CNN、LP3/LP4为配对CE；Xu-Full为原matched模式、paper_sum重构与DML各权重1；B0为配对CE+0.2恢复项，B1为配对CE+1.0原复谱/相强度/噪声标定恢复项。
- B0、B1各有10轮原恢复预训练，其他模型无额外预训练；这项预算差异保留。
- 实测：仍为88个Dev编号，每条件2474周期，条件为Clean、10/15/20/25/30/35/40/5/0/−5/−10 dB，共504组评价。所有源训练完成并冻结best后才评价。
- 不读取347个Final-Holdout波形；Dev已用于此前方法开发，本轮结果不作为最终盲测。
- 结果写入新目录，保存best、终轮权重、每轮F1/CE/损失分项、逐周期预测与所有种子，原实验保持原字节。

配置：[source_cycle_benchmark_v1.json](../configs/source_cycle_benchmark_v1.json)。运行入口：[source_cycle_benchmark.py](../scripts/source_cycle_benchmark.py)。

```bash
python scripts/source_cycle_benchmark.py train --methods resnet1d --seeds 42 --source-root /path/to/source_cycle_split_v2 --out /path/to/new_results
python scripts/source_cycle_benchmark.py evaluate --source-root /path/to/source_cycle_split_v2 --snr-root /path/to/author_snr --strong-root /path/to/generated/target --extreme-root /path/to/extreme_dev --out /path/to/new_results
python scripts/source_cycle_benchmark.py summarize --out /path/to/new_results
```

训练loss同时记录各batch等权均值与按样本数加权均值；更新次数为每轮31次，最后一批2个基础周期、配对后4个输入。Xu的DML可为负，不把总loss为负直接判为运行失败。
