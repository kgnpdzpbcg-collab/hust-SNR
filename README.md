# hust-SNR

HUST五相IM-AA数据的实验配置、指标记录、数据解析及波形图。

- [实验配置与完整结果](docs/EXPERIMENT_RECORD.md)
- [数据字段及解析](docs/DATA_PARSING.md)
- [六类波形图](docs/WAVEFORMS.md)
- [逐种子结果CSV](records/gate0_v1/PER_SEED_RESULTS.csv)
- [冻结配置](configs/protocol_gate0_v1.json)
- [数据划分清单](configs/splits_gate0_v1.json)

## 本轮指标

Macro-F1（%，种子42/43/44的均值 ± 样本标准差）。

| 训练条件 | 实测Dev Clean | 实测Dev 10 dB |
|---|---:|---:|
| baseline | 96.77 ± 2.55 | 78.50 ± 7.57 |
| augmentation | 95.08 ± 1.71 | 83.73 ± 7.25 |

本轮使用88条实测开发记录、每条件2474个周期。347条Final-Holdout未评估。

## 仓库内容

- `scripts/`：实际运行的预处理、CNN训练、核验代码，及只读数据审计/波形绘图代码。
- `configs/`：本轮冻结参数与按原记录分组的划分。
- `records/gate0_v1/`：训练历史、逐条预测、混淆矩阵、指标、权重哈希和核验记录。
- `records/data_audit_v1/`：文件/标签元数据、SNR配对及重复关系。
- `figures/waveforms/`：12张波形图的PNG/SVG/PDF，共36个图文件及来源清单。

原始MAT/RAR、模型权重、环境和中间电流数组未上传。波形图对应冻结Train/Dev中的示例；图中记录编号和周期边界在来源清单中列出。

## 数据来源

数据源：[Multiphase converter switching-device open-circuit fault dataset](https://github.com/XHX-HUST/Multiphase-converter-switching-device-open-circuit-fault-data-set)。
配套论文：*Fully Simulated Data-Driven Domain Generalized Method for Multiphase Converters Fault Diagnosis*，IEEE TPEL 40(1):2099–2113，DOI：10.1109/TPEL.2024.3461159。

本仓库文档记录已执行配置、数值和文件状态。

## V1 噪声一致性实验（新研究分支）

- [V1 实验设计与运行说明](docs/V1_NOISE_CONSISTENCY_PROTOCOL.md)
- [V1 训练和七档 SNR 评估代码](scripts/v1_noise_consistency.py)
- [V1 CPU 单元测试](tests/test_v1_consistency.py)

**状态：提交 `69e2926` 已在原GPU服务器完成三种方法×三个种子的50轮训练，以及Clean＋七档SNR的72次Dev评估；347条Final-Holdout未评估。**

- [V1服务器完整结果](records/v1_noise_consistency/server_69e2926/results/SERVER_RESULTS.md)
- [V1逐种子结果CSV](records/v1_noise_consistency/server_69e2926/results/PER_SEED_RESULTS.csv)
- [V1独立核验记录](records/v1_noise_consistency/server_69e2926/results/VERIFICATION.json)

- [M1固定权重10dB诊断：分类、预处理与配对波形](docs/M1_SNR10_DIAGNOSTIC_RECORD.md)

## Dev 原始 SNR 数据导出

- [只导出88条Dev记录的脚本](scripts/export_snr_data.py)
- [原服务器运行方法及文件结构](docs/SNR_DATA_EXPORT.md)
- [导出脚本的MAT配对与格式测试](tests/test_export_snr_data.py)

在含有原始 SNR MAT 数据的服务器上运行 `python scripts/export_snr_data.py --snr-root /path/to/SNR_parent`。输出位于 gitignored 的 `outputs/dev_snr_export/`，仅包含冻结的88条目标开发记录及可选Clean版本，不接触347条Final-Holdout，也不向GitHub公开原始电流。

## V2 PaiFilter-inspired harmonic shaping (研究分支)

- [V2架构、论文来源及冻结实验协议](docs/V2_PAIFILTER_PROTOCOL.md)
- [V2共享谐波整形与原CNN模型](scripts/v2_paifilter.py)
- [V2源域训练及88条Dev七档SNR测试](scripts/v2_experiment.py)
- [V2模型测试](tests/test_v2_paifilter.py)；[数据流程测试](tests/test_v2_experiment.py)

状态：运行提交`4df46a3`已在原GPU服务器完成三种方法×三个种子的50轮训练，以及Clean＋七档SNR的72次Dev评估。17项测试、36组源Val分数及72组Dev预测复现核验通过；347条Final-Holdout未评估。该实现是对FilterNet PaiFilter频域乘法思想的**任务适配**，不是原论文模型复现。

- [V2服务器实验完整记录](records/v2_paifilter/server_4df46a3/SERVER_RESULTS.md)
- [V2逐种子72行结果](records/v2_paifilter/server_4df46a3/results/PER_SEED_RESULTS.csv)
- [V2独立核验与权重哈希](records/v2_paifilter/server_4df46a3/results/VERIFICATION.json)

## V4 HPASF V0（四组公平频域对照）

已实现轻量 A′＋HPASF V0 的模型和正式训练、Dev评估入口（运行提交`b4801df`已完成正式GPU对照实验）。冻结 V2 原 CNN、整记录噪声增强、源域 checkpoint 选模与 88 条 Dev；不接触 347 条 Final-Holdout。

- [V4 模型：第4阶静态／噪声感知补偿](scripts/v4_hpasf.py)
- [V4 训练、Dev评估、汇总](scripts/v4_experiment.py)
- [V4 完整设计与实验协议](docs/V4_HPASF_PROTOCOL.md)
- [模型测试](tests/test_v4_hpasf.py) · [训练流程和汇总测试](tests/test_v4_experiment.py)

```bash
python scripts/v4_hpasf.py
python scripts/v4_experiment.py smoke
python -m pytest -q tests/test_v4_hpasf.py tests/test_v4_experiment.py
python scripts/v4_experiment.py train
python scripts/v4_experiment.py evaluate --snr-root /path/to/SNR_parent
python scripts/v4_experiment.py summarize
```

四组：`lowpass3`、`lowpass4`、`static4`、`hpasf`；后三组均以 V2 的固定前三阶低通为对照，只有第4阶的增益处理不同。V4 需重新训练 `lowpass3`，才能与其他三组公平比较。

- [V4服务器完整结果](records/v4_hpasf/server_b4801df/SERVER_RESULTS.md)
- [V4逐种子96行结果](records/v4_hpasf/server_b4801df/results/PER_SEED_RESULTS.csv)
- [V4独立核验](records/v4_hpasf/server_b4801df/results/VERIFICATION.json)

## 本项目派生5/0 dB强噪声数据

- [生成方式、数量、核验和源验证成绩](records/strong_snr_v1/server_20261012/GENERATION_RECORD.md)
- [独立生成脚本](scripts/generate_strong_snr.py) · [扩展数据配置](configs/strong_snr_data_v1.json)

服务器已生成998个新MAT：仿真59个编号×两档、实测440个编号×两档。实测由作者10 dB噪声放大派生；旧V4未重训，实测最终保留集没有进行模型评估。

## V4固定权重5/0 dB实测压力测试

运行入口提交`84f6757`，固定原12份最佳权重，复用原服务器环境；同一88个实测编号、每档2474个周期。完成24组新5/0 dB评估及12组10 dB重放；没有训练或更新模型。36组预测和指标独立核查通过，347个最终保留编号未评估。

- [完整客观结果、六类F1/Recall、漏报和配对差](records/v4_strong_snr/server_84f6757/SERVER_RESULTS.md)
- [实际使用的88个数据文件及周期数量](records/v4_strong_snr/server_84f6757/EVALUATED_RECORDS.csv)
- [独立推理入口](scripts/evaluate_v4_strong_snr.py) · [统计和数据池测试](tests/test_evaluate_v4_strong_snr.py)
- [逐种子结果](records/v4_strong_snr/server_84f6757/results/PER_SEED_RESULTS.csv) · [独立核查](records/v4_strong_snr/server_84f6757/results/VERIFICATION.json)

## V4扩大仿真噪声后重新训练

运行提交`ddb0638`，独立协议`V4_strong_noise_v1`。训练每轮随机抽取0/5/10/15/20/25/30/35/40 dB，固定仿真验证使用0/5/10/20/30 dB。四种模型、三个种子均从原随机初始化完整训练50轮；模型、损失、更新预算和数据划分不变。已在原服务器环境完成12组训练及原88个实测编号的Clean＋九档SNR共120组评估。

- [完整客观结果及新旧配对差](records/v4_strong_retrain/server_ddb0638/SERVER_RESULTS.md)
- [独立重训入口](scripts/v4_strong_noise_experiment.py) · [实验配置](configs/v4_strong_noise_v1.json) · [协议测试](tests/test_v4_strong_noise_experiment.py)
- [120组逐种子新旧对照](records/v4_strong_retrain/server_ddb0638/results/BEFORE_AFTER_PER_SEED.csv)
- [训练、选模和预测核查](records/v4_strong_retrain/server_ddb0638/results/VERIFICATION.json)
- [实际使用的34/15个仿真编号](records/v4_strong_retrain/server_ddb0638/SOURCE_RECORDS.csv) · [88个实测编号](records/v4_strong_retrain/server_ddb0638/TARGET_RECORDS.csv)

37项测试、72组最佳权重仿真验证指标和120组实测预测复现核验通过。旧结果及347个最终保留编号的评估状态保持不变。


## V5 NA-HSF（Noise-Aware Harmonic Selective Fusion）

在原 25,620 参数 A′ 1D-CNN 前加入**噪声感知谐波选择融合模块**；仅学习五相共享的第4阶增益，新增一个 3→8→1 的 41 参数门控。训练使用原 V4 的 10–40 dB 源域增强、10/20/30 dB 源域验证选模、50 epochs、batch32、Adam 1e−3、三种子 42/43/44。5/0 dB 为**测试阶段的人工合成压力条件**；不接触347条 Final-Holdout。**运行提交ae2716b已完成三种子GPU训练及30组Dev评估，36项测试和独立复现核查通过。**

- [数学定义和冻结实验协议](docs/V5_NA_HSF_PROTOCOL.md)
- [NA-HSF 网络](scripts/v5_nahsf.py) · [训练、评估及 V4 配对对比](scripts/v5_experiment.py)
- [频域模块测试](tests/test_v5_nahsf.py) · [源域协议与历史结果测试](tests/test_v5_experiment.py)

在仓库根目录：

    git fetch origin
    git switch research/na-hsf-v5
    python scripts/v5_nahsf.py
    python scripts/v5_experiment.py smoke
    python -m pytest -q tests/test_v5_nahsf.py tests/test_v5_experiment.py
    python scripts/v5_experiment.py train
    python scripts/v5_experiment.py evaluate --snr-root "/path/to/original_SNR_parent"
    python scripts/v5_experiment.py evaluate --strong-snr-root "/path/to/generated/target" --conditions 5 0
    python scripts/v5_experiment.py summarize --conditions clean 10 15 20 25 30 35 40 5 0

默认结果存放 outputs/v5_nahsf/；汇总使用已有 V4 的 LP3/LP4 历史逐种子预测进行严格配对，并记录门控随噪声变化的均值与分位数。

- [V5服务器完整结果、门控统计和历史低通配对](records/v5_nahsf/server_ae2716b/SERVER_RESULTS.md)
- [逐种子30组成绩](records/v5_nahsf/server_ae2716b/results/PER_SEED_RESULTS.csv) · [60行历史配对](records/v5_nahsf/server_ae2716b/results/PAIRED_V4_COMPARISON.csv)
- [180行六类指标](records/v5_nahsf/server_ae2716b/results/PER_CLASS_PER_SEED.csv) · [门控统计](records/v5_nahsf/server_ae2716b/results/GATE_PER_SEED.csv)
- [独立核查](records/v5_nahsf/server_ae2716b/results/VERIFICATION.json)

实测Macro-F1（三种子均值±样本标准差）：Clean 95.75±6.38%，10 dB 95.26±6.53%，5 dB 93.56±6.54%，0 dB 87.53±7.40%。源验证选择的最佳轮次为3、4、8；旧V4原始/压力/重训结果保持不变。

## 四模型统一重跑：训练10–40 dB，验证增加0/5 dB

运行提交`56513c7`。四模型为A′无滤波、三阶低通、四阶低通及V5 NA-HSF；各完成三个种子50轮训练和原88个实测编号的十条件评估。训练仍为10–40 dB，验证扩展为Clean及0/5/10/20/30 dB。模型、数据划分、初始化、训练随机流、损失、优化器和Clean容差不变。

600轮训练loss及旧规则对应的12份权重与原实验逐数值/逐张量一致。旧三档平均验证达到100%的轮数为67/600，新五档平均为0/600。41项测试、72组源验证指标和120组实测预测独立核查通过；最终保留编号未评估。

- [完整客观记录：验证饱和、四模型及新旧选模对照](records/four_models_expanded_val/server_56513c7/SERVER_RESULTS.md)
- [独立统一入口](scripts/four_models_expanded_val.py) · [冻结协议](configs/four_models_expanded_val_v1.json) · [测试](tests/test_four_models_expanded_val.py)
- [验证饱和统计](records/four_models_expanded_val/server_56513c7/results/VALIDATION_SATURATION.csv)
- [训练轨迹及旧权重审计](records/four_models_expanded_val/server_56513c7/results/TRAINING_TRAJECTORY_PARITY.json)
- [120组逐种子结果](records/four_models_expanded_val/server_56513c7/results/PER_SEED_RESULTS.csv) · [720行类别指标](records/four_models_expanded_val/server_56513c7/results/PER_CLASS_PER_SEED.csv)
- [独立核验](records/four_models_expanded_val/server_56513c7/results/VERIFICATION.json)

## 固定α=0.5融合与V5对照

运行提交`3c38037`。新增固定α模型三个种子各50轮训练，不使用三维频谱描述符或Gate MLP；第四阶谐波增益恒定0.5，CNN正常训练。复用上一轮V5三个种子的五档源验证选模结果，不重训V5。CNN、训练10–40 dB噪声、种子、损失、预算及Clean+0/5/10/20/30 dB源验证选模规则相同。固定模型最佳轮次为20、7、15，V5为8、7、6。

实测Dev Macro-F1（三种子均值±样本标准差，%）：

| 条件 | 固定α=0.5 | V5 NA-HSF |
|---|---:|---:|
| Clean | 95.62 ± 4.94 | 98.22 ± 1.51 |
| 10 dB | 93.85 ± 5.55 | 97.51 ± 1.33 |
| 5 dB | 92.05 ± 6.43 | 96.08 ± 0.76 |
| 0 dB | 87.45 ± 8.22 | 90.97 ± 1.33 |

三个种子并发训练/评估/汇总20.940秒。47项测试、36组源验证与60组两模型实测预测复现、30行配对及1412个旧文件SHA核验通过；347个最终保留编号未评估。

- [完整客观实验记录及逐种子成绩](records/fixed_half_fusion/server_3c38037/SERVER_RESULTS.md)
- [固定融合模型](scripts/fixed_half_fusion.py) · [独立入口](scripts/fixed_half_experiment.py) · [冻结配置](configs/fixed_half_vs_v5_v1.json) · [协议](docs/FIXED_HALF_VS_V5_PROTOCOL.md)
- [逐种子配对](records/fixed_half_fusion/server_3c38037/results/PAIRED_V5_COMPARISON.csv) · [六类指标](records/fixed_half_fusion/server_3c38037/results/PER_CLASS_PER_SEED.csv)
- [具体数据编号](records/fixed_half_fusion/server_3c38037/results/DATA_FILE_INDEX.csv) · [独立核验](records/fixed_half_fusion/server_3c38037/results/VERIFICATION.json)

## 五模型新增种子2024、2025、2026

运行提交`3f07228`，新增五模型×三个种子共15次50轮训练，训练/验证数据、10–40 dB训练噪声、五档源验证选模、CNN及损失保持冻结。原42/43/44结果复用，分别统计原三个、新三个及合并六个种子。15次训练、150组新增评估与合并汇总墙钟时间83.170秒。

全部六种子的实测Dev Macro-F1（均值±样本标准差，%）：

| 模型 | Clean | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|
| A′无滤波 | 90.83 ± 2.62 | 86.18 ± 3.79 | 75.99 ± 4.29 | 53.72 ± 4.05 |
| 三阶低通 | 94.68 ± 2.89 | 93.78 ± 2.96 | 92.16 ± 3.32 | 86.83 ± 4.36 |
| 四阶低通 | 95.52 ± 0.93 | 94.46 ± 1.59 | 92.45 ± 2.45 | 86.32 ± 3.78 |
| V5 NA-HSF | 94.34 ± 6.43 | 93.32 ± 6.68 | 91.18 ± 7.58 | 85.49 ± 7.80 |
| 固定α=0.5 | 94.42 ± 3.68 | 93.01 ± 3.95 | 90.53 ± 4.82 | 85.03 ± 6.39 |

54项测试、90组新源验证与150组新实测预测复现、300行逐种子指标、1800行类别指标和600行模型配对核验通过。30个组合源验证选模独立重算，1462个原文件SHA不变；347个Final-Holdout编号未评估。

- [完整客观报告：新增、原三种子及六种子汇总](records/five_models_extra_seeds/server_3f07228/SERVER_RESULTS.md)
- [300行逐种子成绩](records/five_models_extra_seeds/server_3f07228/results/PER_SEED_RESULTS.csv) · [600行配对](records/five_models_extra_seeds/server_3f07228/results/PAIRED_PER_SEED.csv) · [1800行类别指标](records/five_models_extra_seeds/server_3f07228/results/PER_CLASS_PER_SEED.csv)
- [独立入口](scripts/five_models_extra_seeds.py) · [冻结配置](configs/five_models_extra_seeds_v1.json) · [协议](docs/FIVE_MODELS_EXTRA_SEEDS_PROTOCOL.md) · [独立核验](records/five_models_extra_seeds/server_3f07228/results/VERIFICATION.json)

## V5种子2025诊断记录

固定权重脚本`edf78ff`、精确回放脚本`d49d9ac`。原第18轮权重在实测Clean为81.90% Macro-F1；OUS(A)、OLS(A)召回率为64.50%、47.14%，两类占266/315个Clean错误。推理强制α=0.5为82.07%，全部BN源数据校准为78.19%；四条件MD低电流分支翻转均为0。

原V5及固定α种子2025各50轮训练轨迹和原最佳权重完全复现。事后Dev扫描记录：V5第4轮源Clean/实测Clean为94.69%/90.53%，第18轮为100%/81.90%，第50轮为87.19%/63.51%。源Clean候选阈值为99%，正式选择仍为第18轮。所有诊断成绩独立保存，不替换正式结果。

- [完整客观诊断：具体文件误判、门控/BN对照和轮次扫描](records/v5_low_seed_diagnosis/server_edf78ff/DIAGNOSIS_RECORD.md)
- [记录级错误](records/v5_low_seed_diagnosis/server_edf78ff/results/PER_RECORD_ERRORS.csv) · [400行轮次扫描](records/v5_low_seed_diagnosis/server_edf78ff/trajectory/EPOCH_TARGET_SCAN.csv)
- [独立核验488组指标及7744行记录错误](records/v5_low_seed_diagnosis/server_edf78ff/VERIFICATION.json)


## Xu 2025 原论文完整 CAE＋DML 复现（独立研究分支）

原论文：Xu et al., *Fully Simulated Data-Driven Domain Generalized Method for Multiphase Converters Fault Diagnosis*, IEEE TPEL 2025, DOI: 10.1109/TPEL.2024.3461159。

过去的 A′ 仅是 25,620 参数 Conv1D 分类器，**并非原论文完整模型**。本分支新增 35,475 参数的完整三层卷积编码器、四层重构解码器、352→50→6 分类头、重构 MSE、类内/类间 DML，以及论文式 (19)–(20) 联合优化；完整模型前接 NA-HSF 则为 35,516 参数。

- [逐公式复现依据及不可消除的论文实现歧义](docs/XU2025_FULL_REPRODUCTION.md)
- [完整 CAE＋DML 模型与损失](scripts/xu2025_full_model.py) · [训练/评估脚本](scripts/xu2025_full_experiment.py)
- [冻结配置](configs/xu2025_full_v1.json) · [网络单元测试](tests/test_xu2025_full_model.py) · [协议测试](tests/test_xu2025_full_experiment.py)
- 两个隔离轨道：`paper_like`（优先复现作者的完整网络及三损失）与 `matched`（统一原 10–40 dB paired source training / 源域0/5/10/20/30 dB选模，公平比较完整 CAE＋DML 与 Full＋LP3/LP4/NA-HSF）。

**状态**：本地 CPU 单元测试和形状/梯度 smoke 已通过；**尚未运行正式服务器完整模型实验**。原论文没有公开完整模型训练源码，表 III 存在通道与 FC 输入维度笔误，也未给出解码器卷积 padding、损失比例与学习率衰减数值；本代码据论文公开内容复现，并在协议中明确披露所有必要假设，不能声称逐字节复现作者原权重与 99.58% 数值。Final-Holdout 仍严格隔离。
