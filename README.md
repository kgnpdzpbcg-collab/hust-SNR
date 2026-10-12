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
