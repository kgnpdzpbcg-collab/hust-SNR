# 五模型追加随机种子协议

新增种子2024、2025、2026；模型为A′无滤波、三阶低通、四阶低通、V5 NA-HSF和固定α=0.5。共新增15次训练，每次50轮。原种子42、43、44结果复用，原四模型运行提交`56513c7`，固定α运行提交`3c38037`。不按成绩排除任何种子，分开统计原三种子、新三种子与合并六种子。

只改变随机种子，因此CNN初始化、训练噪声实现和训练批次顺序随种子改变；数据划分、验证噪声实现和实测输入保持固定。五模型在同一种子下使用完全相同的初始CNN、训练噪声和批次顺序。

共同设置：仿真训练34记录738周期，验证15记录302周期；五相4096点完整周期，原MD归一化、128点CNN输入；batch32，Adam学习率0.001、无权重衰减；损失为`0.5 CE(clean)+0.5 CE(noisy)`。每轮训练噪声10/15/20/25/30/35/40 dB，固定验证为Clean及0/5/10/20/30 dB。选模仅用源验证：Clean距峰值不超过1个百分点，再依次按五档平均Macro-F1、Clean、最早轮次排序。

模型实现不变，独立入口`scripts/five_models_extra_seeds.py`调用原共同训练/评估函数。旧配置和入口不改写。原CNN参数25,620；V5增加41门控参数；固定α保持第四阶增益0.5，没有描述符、Gate MLP或学习滤波参数。

目标评估沿用88个实测Dev编号，每条件2474周期；条件为Clean、10–40 dB七档、5/0 dB。后两档来自作者10 dB噪声残差放大，是人工合成压力条件，不是独立采集记录。仿真Source-Test和347个Final-Holdout编号不评估。目标标签仅用于事后统计，不用于训练、选模或微调。

服务器复用既有uv/PyTorch环境，新运行目录单独建立，拒绝覆盖模型/评估文件。最多四个训练/评估进程并发，每进程四条CPU线程；故障保留日志，不自动换配置重试。

运行示例：

    python scripts/five_models_extra_seeds.py train --output /new/run/results --strong-data-root /existing/strong_data
    python scripts/five_models_extra_seeds.py evaluate --output /new/run/results --strong-data-root /existing/strong_data --snr-root /existing/target_snr
    python scripts/five_models_extra_seeds.py summarize --output /new/run/results --four-results /existing/four/results --fixed-results /existing/fixed/results

支持`--methods`和`--seeds`拆分运行，实际各阶段命令、进程和耗时写入服务器日志。统计单位为随机种子；均值和样本标准差ddof=1，同时保存全部逐种子成绩。十对模型的差值按同种子先相减，再汇总；不给出自动显著性结论。

预期输出：15份新模型训练历史、150组新实测预测；合并两次历史实验后300行逐种子指标、1800行类别指标、600行逐种子配对、原/新/六种子汇总及V5门控统计。独立核验包含源域选模重算、最佳权重重载预测和sklearn指标复算、代码/旧文件SHA。此协议本身不代表任务已经运行或验证完成。
