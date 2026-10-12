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

## V3 CAE + Low-pass Prior

论文HUST原有CAE-Classification含Encoder、Decoder与分类器；之前V1/V2 CNN实际上沿用了同一Encoder+Classifier，本分支补充Decoder重构MSE并增加低通先验学习频谱增益（128个参数）。保留基准比较：`cae_identity`、`cae_lowpass3`、`cae_prior`、`cae_prior_no_rec`。

- [CAE与低通先验模型](scripts/v3_cae_prior.py)
- [复用V2源域训练和Dev评估的V3命令行](scripts/v3_experiment.py)
- [V3实验协议、论文Table III对齐说明](docs/V3_CAE_PRIOR_PROTOCOL.md)
- [V3模型测试](tests/test_v3_cae_prior.py)；[V3运行入口测试](tests/test_v3_experiment.py)

在服务器执行 `python scripts/v3_experiment.py smoke`，随后 `python scripts/v3_experiment.py train`、`python scripts/v3_experiment.py evaluate --snr-root /PATH/TO/SNR_PARENT`、`python scripts/v3_experiment.py summarize`。正式V3 GPU训练和目标Dev结果尚未执行，Final-Holdout没有使用。V1/V2输出不改动。
