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


## Xu 2025 完整 CAE＋DML 原论文基线（单模型补充）

论文：Xu et al., *Fully Simulated Data-Driven Domain Generalized Method for Multiphase Converters Fault Diagnosis*, IEEE TPEL 2025, DOI: 10.1109/TPEL.2024.3461159。

**当前研究方法始终是既有 A′＋NA-HSF（V5），不会替换为 Xu-Full 主干。** A′ 是 25,620 参数的简化 CNN，而 Xu-Full 是具有编码器、解码器、分类器、重构损失和 DML 的完整论文模型（本实现 35,475 参数）。补充完整 Xu-Full 仅为了给现有方法提供忠实的原论文方法对照，并检验在统一源域与强噪声评估协议下的表现。

- [论文公式、层级复现依据、尚未披露的细节与严谨性边界](docs/XU2025_FULL_REPRODUCTION.md)
- [Xu-Full 完整 CAE＋DML 实现](scripts/xu2025_full_model.py) · [独立训练评估入口](scripts/xu2025_full_experiment.py)
- [服务器一键运行：只训练/评估 `matched xu_full`，三个种子](scripts/run_xu2025_full_server.sh)
- [冻结配置](configs/xu2025_full_v1.json) · [模型测试](tests/test_xu2025_full_model.py) · [协议测试](tests/test_xu2025_full_experiment.py)

仅需要运行一个模型类型 `xu_full`；复用已冻结的 34/15 Source-Train/Val、10–40 dB 源训练噪声、Clean+0/5/10/20/30 dB 源验证、42/43/44 三种子、88 记录 Target-Dev 测试。旧 A′、LP3、LP4、固定 α=0.5、V5 及其已有结果全部不动。**不执行 Xu-Full＋LP3/LP4/NA-HSF** 等模型扩展。

`matched` 轨道与既有 V5 使用相同训练和选模协议，但**不同于原作者发表时的训练协议**；`paper_like` 是可选且单独标记的忠实复现参考，不在默认脚本执行。两者不能混称。

**状态：运行提交`bb2f219`已完成matched轨道Xu-Full三个种子GPU训练及30组实测Dev评估。** 10项测试、18组源验证及30组目标预测独立复现通过；最终保留集未评估。原论文部分网络细节和超参有未公开或不一致之处，具体实现假设见文档；本轮使用默认`paper_sum`重构损失，重构与DML权重均1。

实测Dev Macro-F1（种子42/43/44均值±样本标准差，%）：Clean 71.56±2.10，10 dB 64.64±1.02，5 dB 53.08±2.21，0 dB 30.11±5.83。源验证选中的轮次为50、50、49。三种子训练、评估及汇总墙钟时间25.934秒。

- [Xu-Full服务器完整客观记录及同三种子历史配对](records/xu2025_full/server_bb2f219/SERVER_RESULTS.md)
- [150轮损失分量](records/xu2025_full/server_bb2f219/results/matched/reports/TRAIN_LOSS_COMPONENTS.csv) · [30行逐种子成绩](records/xu2025_full/server_bb2f219/results/matched/reports/FULL_PER_SEED_RESULTS.csv)
- [六类指标](records/xu2025_full/server_bb2f219/results/matched/reports/PER_CLASS_PER_SEED.csv) · [独立核验](records/xu2025_full/server_bb2f219/results/matched/reports/VERIFICATION.json)

## 经典方法对照：SVM-RBF、Random Forest、1D-ResNet

**研究分支：`research/classical-baselines-v1`。** 与既有 A′／LP3／LP4／固定α=0.5／V5 NA-HSF／Xu-Full 独立，不修改或重跑历史模型。新增三种标准方法，使用原先冻结的 Source-Train／Val 和 88 条 Target-Dev，且仅使用 seed 42、43、44：

- [三种基线的完整协议及特征说明](docs/CLASSICAL_BASELINES_V1.md) · [冻结配置](configs/classical_baselines_v1.json)
- [SVM/RF 时频特征与 1D-ResNet 模型](scripts/classical_baseline_models.py) · [训练、验证、评估及汇总](scripts/classical_baseline_experiment.py)
- [服务器执行脚本](scripts/run_classical_baselines_server.sh) · [单元测试](tests/test_classical_baselines.py)

SVM 与 RF 使用相同的五相电流120维时域／谐波特征以及源域7次噪声增强视图；ResNet 使用标准残差块与 V5 相同的 MD→128点预处理，在 10–40 dB 配对源训练下进行50轮优化。所有超参数和 checkpoint **只能使用源域 Clean＋0/5/10/20/30 dB 验证选择**；Target-Dev 只能最终推理汇报。5/0 dB 属于合成压力条件，不能称作额外采集的实测数据；347 条 Final-Holdout 不读取。

服务器根目录：

```bash
export SOURCE_STRONG_ROOT="/path/to/strong_data_root"
export TARGET_SNR_ROOT="/path/to/original_SNR_parent"
export TARGET_STRONG_ROOT="/path/to/strong_data_root"
bash scripts/run_classical_baselines_server.sh
```

**状态：运行提交`4860c77`已完成三方法×三个种子的服务器训练及十条件Dev评估。** 11项测试、9组源选模、54组源验证、90组实测预测和3组源训练Scaler复现通过；1951个旧文件SHA不变，最终保留集未读取。四进程并发训练、评估及汇总99.734秒，复用原uv/PyTorch环境。

实测Dev Macro-F1（42/43/44均值±样本标准差，%）：

| 方法 | Clean | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|
| RBF-SVM | 83.55 ± 0.54 | 74.32 ± 1.92 | 21.16 ± 0.59 | 2.98 ± 0.00 |
| Random Forest | 98.83 ± 0.35 | 96.73 ± 0.48 | 86.48 ± 2.37 | 13.05 ± 0.55 |
| ResNet1D | 98.88 ± 1.94 | 98.49 ± 1.35 | 93.35 ± 2.41 | 70.96 ± 9.04 |

- [三方法完整服务器记录、十条件及同三种子历史比较](records/classical_baselines/server_4860c77/SERVER_RESULTS.md)
- [90行逐种子成绩](records/classical_baselines/server_4860c77/results/reports/PER_SEED_RESULTS.csv) · [540行六类指标](records/classical_baselines/server_4860c77/results/reports/PER_CLASS_PER_SEED.csv) · [九方法共同种子表](records/classical_baselines/server_4860c77/results/reports/COMMON_SEED_NINE_METHODS.csv)
- [按文件/类别/时间段统计的实际输入索引](records/classical_baselines/server_4860c77/results/reports/DATA_FILE_INDEX.csv) · [独立核验](records/classical_baselines/server_4860c77/results/reports/VERIFICATION.json)

## 六模型固定权重t-SNE与原空间指标

运行提交`f398d30`。A′无滤波、三阶低通、四阶低通、固定α=0.5、V5及Xu-Full统一采用共同种子42/43/44，读取冻结实测Dev的88个编号、每条件全部2474个周期。使用原源验证最佳权重，未重新训练或选模。Clean和10 dB分别提取编码器352维及分类隐藏层50维，共72组固定参数投影、12张六面板图；全部种子均发布。

36组原预测、72组特征分类头重放、178128行坐标标签及24组指标汇总独立核查通过；重复一组投影的坐标最大绝对差为0。PNG为600 dpi、PDF嵌入字体；二维投影不等于分类成绩，原空间指标另存。最终保留集未评估，原始电流、高维特征及权重未上传。

- [完整客观记录、两层原空间指标及全部图像](records/six_model_tsne/server_f398d30/VISUALIZATION_RECORD.md)
- 主图种子42：[Clean PNG](figures/six_model_tsne/server_f398d30/encoder352_clean_seed42.png) · [10 dB PNG](figures/six_model_tsne/server_f398d30/encoder352_snr10_seed42.png) · [Clean PDF](figures/six_model_tsne/server_f398d30/encoder352_clean_seed42.pdf) · [10 dB PDF](figures/six_model_tsne/server_f398d30/encoder352_snr10_seed42.pdf)
- [72组逐种子原空间指标](records/six_model_tsne/server_f398d30/results/FEATURE_SEPARABILITY_PER_SEED.csv) · [完整汇总](records/six_model_tsne/server_f398d30/results/SUMMARY.json) · [独立核验](records/six_model_tsne/server_f398d30/results/VERIFICATION.json)
- [冻结协议](docs/SIX_MODEL_TSNE_PROTOCOL.md) · [提取与投影入口](scripts/six_model_tsne.py) · [绘图入口](scripts/plot_six_model_tsne.py) · [图像核查](figures/six_model_tsne/server_f398d30/VISUAL_QA.json)

## 七模型单种子t-SNE：补0/5 dB及ResNet

运行提交`d8194e8`。按用户指定仅选V5展示种子；固定规则为共同42/43/44中V5的Clean/10/5/0 dB平均Macro-F1最高，选中44（96.49%）。全部七模型统一使用44的原源验证最佳权重，正式三种子成绩不改变。该选择是使用Dev分数的事后展示选择，不代表三种子平均或稳定性。

原六模型Clean/10 dB的12份坐标原字节复用；新增六模型0/5 dB和ResNet四条件共16份投影，总计28份、四张七面板图。前六模型352维编码特征，ResNet128维池化特征；每条件全部2474周期。28组特征/几何指标、69272行坐标标签及重复ResNet 0 dB投影核查通过，重复坐标差为0；四张图已检查。

- [展示种子选择、七模型成绩及全部PNG/PDF](records/seven_model_tsne/server_d8194e8/VISUALIZATION_RECORD.md)
- [5 dB PNG](figures/seven_model_tsne/server_d8194e8/encoder_snr5_seed44.png) · [0 dB PNG](figures/seven_model_tsne/server_d8194e8/encoder_snr0_seed44.png) · [5 dB PDF](figures/seven_model_tsne/server_d8194e8/encoder_snr5_seed44.pdf) · [0 dB PDF](figures/seven_model_tsne/server_d8194e8/encoder_snr0_seed44.pdf)
- [Clean PNG](figures/seven_model_tsne/server_d8194e8/encoder_clean_seed44.png) · [10 dB PNG](figures/seven_model_tsne/server_d8194e8/encoder_snr10_seed44.png) · [28组原空间指标](records/seven_model_tsne/server_d8194e8/results/FEATURE_SEPARABILITY.csv) · [独立核验](records/seven_model_tsne/server_d8194e8/results/VERIFICATION.json)
- [冻结协议](docs/SEVEN_MODEL_TSNE_PROTOCOL.md) · [特征及投影入口](scripts/seven_model_tsne.py) · [绘图入口](scripts/plot_seven_model_tsne.py) · [图像核查](figures/seven_model_tsne/server_d8194e8/VISUAL_QA.json)


## 无残差 Plain CNN1D（与 ResNet 的同深度对照）

研究分支 `research/plain-cnn1d-control-v1`。新增 **Plain CNN1D**，只训练该模型的 seed 42/43/44；历史 A′、V5、ResNet1D、RF/SVM、Xu-Full 及已经上传的 t-SNE 一律不修改或重跑。

- [PlainCNN1D 网络](scripts/plain_cnn1d_baseline.py)｜[独立源域训练、Source-Val选模和Target-Dev评估](scripts/plain_cnn1d_experiment.py)
- [服务器脚本](scripts/run_plain_cnn1d_server.sh)｜[冻结配置](configs/plain_cnn1d_control_v1.json)｜[完整协议与学术解释边界](docs/PLAIN_CNN1D_CONTROL_V1.md)｜[模型及损失测试](tests/test_plain_cnn1d_control.py)

结构参照已完成的 1D-ResNet：相同 MD 128 点输入、5→32 stem、32/64/128 通道与每阶段两个双卷积块、GAP 128→6；**去掉残差 shortcut**。其余训练预算、配对噪声、验证选模与原 ResNet 一致，但 shortcut 投影参数也随之去掉，因此**参数量不完全相同**。测试会如实比较两者，绝不能仅因 ResNet 的结果更好就隐藏其结果。

服务器（原项目根目录）：
```bash
export SOURCE_STRONG_ROOT="/path/to/strong_data_root"
export TARGET_SNR_ROOT="/path/to/original_SNR_parent"
export TARGET_STRONG_ROOT="/path/to/strong_data_root"
bash scripts/run_plain_cnn1d_server.sh
```

**状态**：原代码提交`5785ae5`已在原服务器完成三个种子的150轮训练及30组十条件实测Dev评估。服务器8项测试、真实GPU梯度、源噪声匹配、3组源选模、18组源验证及30组目标预测重放通过；2148个旧文件SHA不变。复用原uv/PyTorch环境，三进程训练/评估/汇总32.046秒，Final-Holdout未评估。

Macro-F1（42/43/44均值±样本标准差，%），ResNet/V5为同三个种子的原成绩：

| 模型 | Clean | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|
| Plain CNN1D | 95.10 ± 6.90 | 95.02 ± 3.41 | 89.33 ± 3.75 | 66.54 ± 3.73 |
| ResNet1D | 98.88 ± 1.94 | 98.49 ± 1.35 | 93.35 ± 2.41 | 70.96 ± 9.04 |
| V5 NA-HSF | 98.22 ± 1.51 | 97.51 ± 1.33 | 96.08 ± 0.76 | 90.97 ± 1.33 |

Plain CNN1D参数231078，历史ResNet241702；选中轮次为43/42/22。全部种子保留，种子43的实测Clean为87.20%。未重新训练或覆盖历史方法，未追加t-SNE或低分诊断。

- [CNN1D完整服务器记录、十条件及历史方法对照](records/plain_cnn1d/server_5785ae5/SERVER_RESULTS.md)
- [30行逐种子结果](records/plain_cnn1d/server_5785ae5/results/reports/PER_SEED_RESULTS.csv) · [180行六类指标](records/plain_cnn1d/server_5785ae5/results/reports/PER_CLASS_PER_SEED.csv)
- [150轮损失及源验证](records/plain_cnn1d/server_5785ae5/results/reports/TRAIN_LOSS_HISTORY.csv) · [逐种子ResNet/V5配对差](records/plain_cnn1d/server_5785ae5/results/reports/PAIRED_RESNET_V5.csv)
- [独立核验](records/plain_cnn1d/server_5785ae5/results/reports/VERIFICATION.json) · [文件/类别/时间段索引](records/plain_cnn1d/server_5785ae5/results/reports/DATA_FILE_INDEX.csv)


## Nine-arm Architecture Sweep：一次性验证加宽、加深、多尺度、残差与 ResNet＋谐波前端

**独立研究分支：`research/architecture-sweep-v1`。** 为提升真实域故障诊断效果而设计，不以参数量轻量化为目标。不修改旧 V5/ResNet/LP3/LP4/Xu-Full/历史 t-SNE 权重和实验记录。

- [完整实验协议、9组消融矩阵、可复现性边界](docs/HUST_ARCH_SUITE_V1.md)
- [9个模型定义、频谱前端、MS Stem、等参数残差开关](scripts/hust_arch_suite_models.py)
- [统一源域训练/选模、Target-Dev十条件评估与汇总](scripts/hust_arch_suite_experiment.py)
- [冻结参数与方法矩阵](configs/hust_arch_suite_v1.json)
- [结构、输入尺寸、GPU无关前反向、参数一致性测试](tests/test_hust_arch_suite.py)
- [9模型×3种子服务器批量执行脚本](scripts/run_hust_arch_suite_server.sh)

九种方法：`v5_w`、`v5_wd_plain`、`v5_wd_res`、`v5_w_ms`、`v5_wd_plain_ms`、`v5_wd_res_ms`、`resnet_lp3`、`resnet_lp4`、`resnet_nahsf`。同一源域34/15记录，训练10–40dB，Clean/0/5/10/20/30dB源验证，42/43/44三种子各50轮；实测开发集10条件，每条件2474周期；5/0dB须标为合成压力条件。**全部 27 次训练和 270 组 Dev 推理仅在真实服务器执行，不使用目标标签选模型，347条Final-Holdout严格隔离**。

从仓库根目录：

```bash
git fetch origin
git switch research/architecture-sweep-v1
export SOURCE_STRONG_ROOT="/path/to/existing/strong_data_root"
export TARGET_SNR_ROOT="/path/to/original_target_SNR_parent"
export TARGET_STRONG_ROOT="/path/to/existing/strong_data_root"
bash scripts/run_hust_arch_suite_server.sh
```

**服务器状态：原提交`5c75bea`已完成27次50轮训练及270组实测Dev评估。** 25项测试、27组源选模、162组源验证、全部目标预测独立重放通过；2196个旧文件SHA不变。复用原环境、四工作进程，训练/评估/汇总209.972秒，最终保留集未评估。

Macro-F1（42/43/44均值±样本标准差，%）：

| 模型 | Clean | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|
| v5_w | 97.00 ± 1.22 | 96.61 ± 1.28 | 95.04 ± 1.54 | 90.84 ± 0.05 |
| v5_wd_plain | 93.36 ± 5.71 | 92.68 ± 6.02 | 90.89 ± 6.57 | 86.41 ± 8.02 |
| v5_wd_res | 92.83 ± 3.27 | 91.97 ± 3.37 | 90.56 ± 4.14 | 86.38 ± 5.24 |
| v5_w_ms | 96.53 ± 1.05 | 95.10 ± 1.23 | 93.32 ± 1.66 | 88.41 ± 2.79 |
| v5_wd_plain_ms | 92.40 ± 1.51 | 91.88 ± 1.89 | 90.44 ± 2.28 | 86.34 ± 3.41 |
| v5_wd_res_ms | 93.32 ± 4.05 | 92.56 ± 4.21 | 91.00 ± 4.26 | 85.59 ± 5.53 |
| resnet_lp3 | 96.10 ± 5.05 | 96.14 ± 4.55 | 95.42 ± 4.87 | 92.38 ± 4.84 |
| resnet_lp4 | 99.42 ± 0.50 | 99.00 ± 0.68 | 98.10 ± 0.85 | 94.44 ± 1.05 |
| resnet_nahsf | 94.80 ± 5.65 | 94.28 ± 5.54 | 93.59 ± 5.68 | 89.84 ± 6.07 |

- [完整服务器记录、逐种子成绩及历史V5/ResNet配对](records/architecture_suite/server_5c75bea/SERVER_RESULTS.md)
- [270行逐种子结果](records/architecture_suite/server_5c75bea/results/reports/PER_SEED_RESULTS.csv) · [1620行六类指标](records/architecture_suite/server_5c75bea/results/reports/PER_CLASS_PER_SEED.csv)
- [1350轮损失和源验证](records/architecture_suite/server_5c75bea/results/reports/TRAIN_LOSS_HISTORY.csv) · [390行配对差](records/architecture_suite/server_5c75bea/results/reports/PAIRED_ARCHITECTURE_COMPARISONS.csv)
- [独立核验](records/architecture_suite/server_5c75bea/results/reports/VERIFICATION.json) · [CUDA核验设置诊断](records/architecture_suite/server_5c75bea/REPLAY_SETTINGS_DIAGNOSIS.json)

## −5 dB派生数据（仅生成与核验）

沿用原5/0 dB加噪方法，在服务器独立生成59个仿真和440个实测MAT，共499份。仿真按整记录每相RMS校准高斯噪声；实测使用作者干净/10 dB配对残差×5.623413251903491。全部逐文件核验通过，原数据、旧0/5 dB文件及模型权重SHA不变；没有重新训练、模型推理或修改冻结实验配置。

- [−5 dB配置](configs/strong_snr_minus5_v1.json) · [生成代码](scripts/generate_strong_snr.py)
- [服务器生成记录、保存路径与核验结果](records/strong_snr_minus5/server_20261014/GENERATION_RECORD.md)
- [499份文件索引](records/strong_snr_minus5/server_20261014/GENERATED_FILE_INDEX.csv) · [独立核验](records/strong_snr_minus5/server_20261014/INDEPENDENT_VERIFICATION.json)

### 九模型固定权重−5 dB评估

服务器已固定原架构实验27份源域选出的权重，在原88个实测Dev编号、2474周期上完成−5 dB推理；没有重新训练、选模或评估Final-Holdout。全部27组0 dB历史预测及27组−5 dB重复预测完全复现；原权重、代码、结果和输入指纹保持不变。

−5 dB Macro-F1（42/43/44均值±样本标准差，%）：v5_w 77.96±0.66、v5_wd_plain 74.50±7.25、v5_wd_res 74.34±7.07、v5_w_ms 77.67±3.93、v5_wd_plain_ms 74.79±3.86、v5_wd_res_ms 71.07±7.73、resnet_lp3 80.24±7.75、resnet_lp4 80.43±2.92、resnet_nahsf 75.69±5.10。

- [独立推理入口](scripts/evaluate_arch_suite_minus5.py) · [冻结评估配置](configs/architecture_minus5_fixed_weights_v1.json)
- [服务器完整结果、逐种子及0/5 dB配对对照](records/architecture_minus5/server_20261014/SERVER_RESULTS.md) · [独立核验](records/architecture_minus5/server_20261014/results/reports/VERIFICATION.json)

## LP4低阶谐波任务难度对照

冻结45维DC＋1–4阶复谐波表示，新增LR、RBF-SVM和1670参数小MLP。沿用34/15源划分、50轮10–40 dB噪声及五档源验证选模，复用原LP4＋A′ CNN/ResNet权重。三个种子全部源域选择核验后，统一完成88条实测Dev的165组评估；没有使用Target调参或Final-Holdout。

实测Clean Macro-F1（42/43/44均值±样本标准差，%）：LR 53.50±0.34、SVM 64.60±0.37、小MLP 58.82±1.69、A′ CNN 96.28±0.46、ResNet 99.42±0.50。−5 dB分别30.46±0.26、2.91±0.00、25.11±1.24、73.23±2.78、80.43±2.92。

- [执行前冻结协议](docs/LP4_TASK_DIFFICULTY_V1.md) · [配置](configs/lp4_task_difficulty_v1.json) · [特征与小MLP](scripts/lp4_difficulty_models.py) · [独立入口](scripts/lp4_difficulty_experiment.py)
- [服务器完整记录、源训练/验证成绩及实测各档对照](records/lp4_task_difficulty/server_20261014/SERVER_RESULTS.md) · [独立核验](records/lp4_task_difficulty/server_20261014/results/reports/VERIFICATION.json)

## 数据类别几何与物理量描述

只读比较原45维复谐波和25维DC/谐波幅值，没有训练分类器或修改前轮参数。当前88条Dev中，25维与最近源训练样本的等类别/等文件标签一致比例为Clean 99.21%、10 dB 97.71%、5 dB 96.35%、0 dB 94.19%、−5 dB 86.88%；这些是描述统计，不是Macro-F1或新的正式Test准确率。

- [分析代码](scripts/analyze_data_geometry.py) · [数值测试](tests/test_data_geometry.py)
- [服务器完整记录、DC/RMS、类对排序与类间/类内距离](records/data_geometry/server_20261014/DATA_ANALYSIS_RECORD.md) · [完整统计](records/data_geometry/server_20261014/results/DATA_GEOMETRY.json)


## B0 / A0 独立抗噪算法原型（服务器实验已完成）

研究分支：research/harmonic-recovery-wavelet-a0-b0。ResNet+LP4仅为历史强基线，不限制新模型结构。

- **B0 多相复谐波恢复**：[模型代码](scripts/hust_b0_harmonic_recovery.py)，FFT保留复谐波0–8阶，噪声代理条件化的可学习增益/复残差修正，源域干净谐波恢复损失＋ResNet分类。额外10轮源域恢复预训练。
- **A0 小波注意力CNN**：[模型代码](scripts/hust_a0_wavelet_attention.py)，无需LP4，五相共享RMS归一化、512点、四层Haar DWA/FAM＋GroupNorm CNN，源域配对CE训练50轮。
- [冻结实验配置](configs/hust_prototypes_a0_b0_v0.json) · [训练、实测Dev评价和汇总入口](scripts/hust_prototypes_experiment.py) · [CPU单元测试](tests/test_hust_prototypes_a0_b0.py)
- [仅88条Dev记录的−5/−10dB数据生成器](scripts/generate_extreme_dev_snr.py) · [服务器一键脚本](scripts/run_hust_prototypes_server.sh)
- **[完整算法定义、张量尺寸、损失、运行说明及限制](docs/HUST_A0_B0_PROTOTYPES_V0.md)**

两模型的测试目标是5/0/−5/−10dB，固定训练仍为源域10–40dB、源验证0/5/10/20/30dB、三种子42/43/44。−5/−10为作者10dB实测残差缩放，**不是新采集**。

原提交`b0f97ce`的GPU预检发现CUDA median与确定性模式冲突；提交`c0b30c9`改为稳定排序后取同一定义的中位数，服务器10项测试通过。算法结构、损失及冻结配置保持不变。已完成6组训练、24组Dev评估及全部预测重放；训练/评估/汇总82.650秒，复用原环境，最终保留集未评估。

Macro-F1（42/43/44均值±样本标准差，%）：

| 模型 | 5 dB | 0 dB | −5 dB | −10 dB |
|---|---:|---:|---:|---:|
| A0 | 69.16 ± 15.12 | 62.58 ± 15.89 | 55.87 ± 17.87 | 47.12 ± 10.54 |
| B0 | 93.72 ± 2.05 | 87.15 ± 8.39 | 66.21 ± 18.00 | 38.33 ± 17.99 |
| 历史LP4＋ResNet | 98.10 ± 0.85 | 94.44 ± 1.05 | 80.43 ± 2.92 | 51.81 ± 3.43 |

- [服务器完整记录、逐种子结果、源选模及CUDA兼容性说明](records/prototypes_a0_b0/server_c0b30c9/SERVER_RESULTS.md)
- [24行逐种子结果](records/prototypes_a0_b0/server_c0b30c9/results/reports/PER_SEED_RESULTS.csv) · [144行逐类指标](records/prototypes_a0_b0/server_c0b30c9/results/reports/PER_CLASS_PER_SEED.csv)
- [300轮分类损失及源验证](records/prototypes_a0_b0/server_c0b30c9/results/reports/TRAIN_LOSS_HISTORY.csv) · [30轮B0恢复预训练](records/prototypes_a0_b0/server_c0b30c9/results/reports/B0_PRETRAIN_RECOVERY_HISTORY.csv)
- [同种子LP4＋ResNet配对](records/prototypes_a0_b0/server_c0b30c9/results/reports/PAIRED_RESNET_LP4.csv) · [独立核验](records/prototypes_a0_b0/server_c0b30c9/results/reports/VERIFICATION.json)

### A0/B0验证与算法实现审计

第一阶段只读9份原权重，重放原24组实测预测；训练/验证编号交集0，302个验证原始周期与训练周期完全重复数0。第二阶段按原配置重放6组训练，300轮历史、30轮B0预训练及六份原best逐张量完全复现，仅增加预先指定的第50轮快照用于诊断。原正式成绩与选模规则保持不变，Final-Holdout未评估。

A0种子42有49轮六版本验证全满分，种子43/44各50轮全满分，原规则选中2/1/1轮。A0实测5 dB原best均值69.16%，epoch50为79.52%；各种子分别60.50→71.83、60.38→96.63、86.62→70.11。B0同条件原best93.72%，epoch50为84.25%。

固定原B0权重，学习恢复后额外去掉5–8阶的诊断结果为−5 dB 81.04±5.27%、−10 dB 55.70±5.71%；原完整forward为66.21±18.00%、38.33±17.99%。这些是固定权重干预，不是重训消融或新的正式模型成绩。

- [完整逐轮过程、结构探针、恢复误差与终轮比较](records/prototypes_audit/server_20261015/AUDIT_RECORD.md)
- [固定权重审计入口](scripts/audit_a0_b0_fixed_weights.py) · [精确训练重放入口](scripts/audit_a0_b0_epoch_replay.py)
- [固定阶段核验](records/prototypes_audit/server_20261015/results/VERIFICATION.json) · [精确重放核验](records/prototypes_audit/server_20261015/epoch_replay/VERIFICATION.json)

### 原正式训练loss检查图

从原日志绘制A0、B0、LP4＋ResNet全部42/43/44种子的50轮训练loss、五噪声源验证F1和原选中轮次，并单列B0额外10轮预训练及联合损失分项。仅作实验检查；没有训练、平滑曲线、虚构验证loss或替换权重。

- [流程、loss定义和原始数值来源](figures/training_checks/a0_b0_lp4_20261015/README.md)
- [训练loss及源验证图](figures/training_checks/a0_b0_lp4_20261015/TRAIN_LOSS_AND_SOURCE_VAL.png) · [B0预训练和loss分项图](figures/training_checks/a0_b0_lp4_20261015/B0_PRETRAIN_AND_LOSS_COMPONENTS.png)


## 新一轮A1 / B1结构化抗噪算法（服务器实验已完成）

研究分支：[research/a1-b1-noise-structure-v1](https://github.com/kgnpdzpbcg-collab/hust-SNR/tree/research/a1-b1-noise-structure-v1)，在已核验A0/B0实验之后新建；**不会改动原A0/B0、ResNet+LP4代码、权重和正式成绩**。

- **A1**：[小波噪声可靠性网络](scripts/hust_a1_reliability_wavelet.py)：五相共享基波幅值标度＋四层Haar DWT，RMS统计预测可衰减的高频门控，低频直通注意力节点。A1仍为无LP4、无ResNet的独立CNN。
- **B1**：[物理相间结构谐波网络](scripts/hust_b1_phase_harmonic.py)：保留0–4阶复谱、基于高频噪声代理和同阶跨相Gram特征进行非放大收缩；源域干净复谱、弱相相对能量和噪声功率提供恢复监督；分类输入包含MD形状与五相共享归一化电流两支证据。
- [独立的source-only训练／实测Dev推理入口](scripts/hust_a1_b1_experiment.py) · [冻结实验配置](configs/hust_prototypes_a1_b1_v1.json) · [单元测试](tests/test_hust_prototypes_a1_b1.py) · [服务器执行脚本](scripts/run_hust_a1_b1_server.sh) · [完整方法及风险说明](docs/HUST_A1_B1_V1.md)

固定源仿真训练10–40dB噪声、三个种子42/43/44、实测Dev评价5/0/−5/−10dB。根据原A0验证F1长时间饱和，新增**仅源验证噪声CE的同F1并列轮次判定**，与旧实验选模规则的区别必须报告。已完成6组GPU训练和24组实测Dev评价，复用原环境，训练/评估/汇总92.666秒；保存权重预测全部重放一致。最终Holdout未评估。

Macro-F1（42/43/44均值±样本标准差，%）：

| 模型 | 5 dB | 0 dB | −5 dB | −10 dB |
|---|---:|---:|---:|---:|
| A1 | 46.38 ± 13.07 | 41.18 ± 9.78 | 35.08 ± 1.43 | 30.73 ± 2.32 |
| B1 | 91.35 ± 6.18 | 87.41 ± 6.49 | 81.68 ± 5.15 | 65.27 ± 6.01 |

- [完整服务器记录及A0/B0/LP4历史对比](records/prototypes_a1_b1/server_dc8cb36/SERVER_RESULTS.md) · [逐种子结果](records/prototypes_a1_b1/server_dc8cb36/results/reports/PER_SEED_RESULTS.csv) · [独立核验](records/prototypes_a1_b1/server_dc8cb36/results/reports/VERIFICATION.json)
- [仿真划分待处理事项（本轮未实施）](docs/DATA_SPLIT_PENDING.md)

## 当前仿真Train/Val：全63文件周期划分V2

2026-10-15按用户要求纳入全部63个仿真MAT，4个Excel遗漏条目按已核对的命名补录并标记来源。全部13组实际工况均参与Train/Val；健康片段去重，各片段按约80/20时间分块，中间隔开1个完整周期。Train为962周期，Val为221周期，另78周期不用；不再单独保留仿真Source-Test。样本长度仍为每相4096点，实测Dev/Holdout边界不变。

A1/B1入口默认使用新划分和独立输出目录；历史结果仍属于旧34/15文件协议。本次只准备及核对数据，没有新训练成绩。

- [完整规则、类别数量、标签来源与命令](docs/SOURCE_CYCLE_SPLIT_V2.md) · [数据配置](configs/source_cycle_split_v2.json)
- [生成及通用读取接口](scripts/source_cycle_split.py) · [重复、隔离和入口选择测试](tests/test_source_cycle_split.py)
