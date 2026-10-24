# V3.13模型：347条Final-Holdout固定权重评价

状态：2026-10-24已在服务器完成。此347条记录已完成本轮固定方案评价；此前文档中的“未评价”是各历史实验当时的状态。

## 固定范围

- 论文：`论文稿/AHR_Net_中文会议论文_V3.13.docx`；读取时SHA256为`65ddd8cfd531a5d81410cfbe9033c4ebc280f66d1f2a0bd323b964f09ab8effb`。
- 主对比：Plain CNN1D、ResNet1D、Xu-Full、ResNet+LP4、HR-Net（B0）、Jung V4、AHR-Net（B1）。
- 五项消融：A1固定LP4＋双视图、A2固定高阶功率统计量、A3去相间谐波特征、A4仅波形形态视图、A5去相间相对幅值损失。
- 种子42、44；噪声5、0、−5、−10 dB；12模型×2种子×4条件，共96组。
- 复用原best.pt，未训练、未重新选轮次，也没有对目标域重标定A2统计量。
- Jung绑定论文实际数值对应的`a0c78d8`单周期V4，未替换成其他版本。

## 数据与执行

- 原冻结划分中的347个保留编号，与88个Dev及其完全重复记录隔离。
- 每档9568周期，六类数量为4294、1149、1159、1166、892、908；全部模型共用相同窗口和标签。
- 仍用单周期：健康0.1–0.45秒；故障从0.5秒后跳过两周期到0.98秒。
- 六个基准/完整模型及五个消融的前端输入为5×4096，Jung按原V4处理为5×128。
- 复用已生成、覆盖全部440编号的四档数据，没有重新生成噪声。
- 服务器：`wang@172.28.128.160`，复用既有uv/PyTorch环境；三个归档代码组在独立进程并发执行。
- 本轮服务器目录：`/home/wang/projects/会议论文/hust/experiments/final_holdout_v313/20261024`。
- 冻结、Dev回放、输入准备、96组正式推理及重放、核验总墙钟时间44.059秒，不含本地开发、部署和下载。

## 核验

- 正式读取Holdout波形前，冻结24份best及原始代码指纹；24组5 dB Dev预测与各自历史记录逐项相同。
- 96组正式预测重放一致；全部Macro-F1、Accuracy及类别指标分别经服务器NumPy和本地Python Counter复算。
- 24份权重、模型参数、BN状态、原代码及原始数据文件指纹未改变。
- 每相整记录SNR与标称档位的最大误差约`3.997e-14 dB`。
- 全部种子保留；没有按Holdout成绩选择种子。两种子样本SD仅描述这两次训练结果，不作为显著性结论。

## 文件

- [两个种子的全部结果及配置](../records/final_holdout_v313/server_20261024/SERVER_RESULTS.md)
- [96行逐种子指标](../records/final_holdout_v313/server_20261024/results/reports/PER_SEED_RESULTS.csv)
- [576行类别指标](../records/final_holdout_v313/server_20261024/results/reports/PER_CLASS_RESULTS.csv)
- [均值及样本SD](../records/final_holdout_v313/server_20261024/results/reports/SUMMARY.csv)
- [24份冻结权重信息](../records/final_holdout_v313/server_20261024/results/reports/WEIGHTS_FROZEN.csv)
- [服务器核验](../records/final_holdout_v313/server_20261024/VERIFICATION.json)
- [本地核验](../records/final_holdout_v313/server_20261024/LOCAL_VERIFICATION.json)
- [评价脚本](../scripts/evaluate_final_holdout_v313.py)；[固定配置](../configs/final_holdout_v313_fixed_weights_20261024.json)

完整逐周期预测及混淆矩阵在记录目录的`results/evaluations/`中。原始电流、权重和输入缓存保留在服务器，不包含在下载记录中；论文正文及旧Dev结果未改写。
