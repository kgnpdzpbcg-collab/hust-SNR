# 固定α=0.5与V5对照协议

本轮只新增固定融合模型三个种子训练。V5复用运行提交`56513c7`的五档源验证选模结果，最佳轮次为8、7、6；不使用旧三档验证结果或训练包含0/5 dB的其他实验。

固定模块输入为五相4096点完整电周期；rFFT采用`norm="ortho"`。直流及1–3阶增益为1，第四阶增益固定0.5，第五阶及以上增益为0，五相共享实数掩码。该操作等价于`0.5×LP3 + 0.5×LP4`。不计算频谱描述符，无Gate MLP，无可学习滤波参数；CNN为原25,620参数网络，按原MD归一化及128点重采样流程训练。V5总参数为25,661。

## 冻结的共同设置

- 仿真训练34个记录、738个周期；仿真验证15个记录、302个周期。数据划分、周期顺序不变。
- 种子42、43、44，各训练50轮；batch32，Adam学习率0.001，无权重衰减。
- 每轮从10/15/20/25/30/35/40 dB抽取仿真训练噪声，随机流与四模型实验一致；不将0/5 dB加入训练。
- 损失为`0.5 CE(clean) + 0.5 CE(noisy)`，每步拼接两种输入执行一次前向计算。
- 固定仿真验证为Clean及0/5/10/20/30 dB；新增0/5使用已核验的源域MAT。
- 先保留Clean Macro-F1距最高值不超过1个百分点的轮次，再按五档噪声平均Macro-F1选模；相同分数按Clean及最早轮次排序。
- 实测评估使用相同88个Dev编号、每条件2474周期，条件为Clean、10–40 dB七档、5/0 dB。5/0来自作者10 dB噪声残差放大，属于人工合成压力条件。
- 无目标域训练、选模或微调；347个最终保留编号及仿真Source-Test不评估。

## 入口和输出

`scripts/fixed_half_experiment.py`为独立入口，显式调用原四模型入口的共同训练和评估函数，不修改原四模型默认列表、网络或配置。`--output`应使用独立新目录，已有模型/评估文件禁止覆盖。`--seeds`支持三个种子分进程运行。训练保留旧三档审计快照，但不用于目标评估。

运行示例（复用服务器已有Python环境）：

    python scripts/fixed_half_experiment.py train --output /new/run/results --strong-data-root /existing/strong_data
    python scripts/fixed_half_experiment.py evaluate --output /new/run/results --strong-data-root /existing/strong_data --snr-root /existing/target_snr
    python scripts/fixed_half_experiment.py summarize --output /new/run/results --v5-results /existing/four_models/results

保存三种子训练历史、选模轮次、30组新预测、两模型60行逐种子指标、360行类别指标、30行配对差值以及均值和样本标准差（ddof=1）。配对差值定义为V5减固定α模型。代码测试及GPU核验、实际运行与独立复现状态记录在服务器结果目录；本协议文件本身不代表实验已完成。
