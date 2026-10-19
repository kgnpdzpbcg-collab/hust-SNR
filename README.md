# hust-SNR

HUST五相IM-AA数据的实验配置、指标记录、数据解析及波形图。

- [Jung Modified CNN与Hang WCNN：当前统一划分、42/44种子及七档结果](docs/JUNG_HANG_CYCLE_V2_RESULTS.md)

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

## 论文对比基线：Jung Modified CNN / Hang WCNN

- [复现方案与运行说明](docs/JUNG_HANG_REPRODUCTION.md)
- [模型结构与五相 VSD/轨迹图](scripts/jung_hang_models.py)
- [训练、Dev 评估与结果汇总入口](scripts/compare_jung_hang.py)
- [CPU 单元测试](tests/test_jung_hang_models.py)

该实现是按论文核心机制进行的近似复现及五相任务适配，不是作者原始代码。模型训练只使用冻结的 Source-Train/Source-Val；目标评估仅限 88 条 Dev 记录，不读取 Final-Holdout。运行结果和权重保存在 gitignored 的 `outputs/jung_hang_comparison/`。

## Jung/Hang统一周期划分实验（2026-10-19已完成）

两模型采用Source-Cycle V2的962/221周期，42/44种子各50轮，10–40 dB配对噪声训练，Clean及0/5/10/20/30 dB源验证，沿用F1后CE平局判据。保留原模型结构及各自128点/轨迹图适配器。四组训练、28组七档实测Dev评价、24组源验证及28组目标预测回放完成。两模型为论文机制的近似复现与任务适配。

- [完整协议、逐种子七档结果](docs/JUNG_HANG_CYCLE_V2_RESULTS.md)
- [十四模型196行合表](records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/ALL_14_MODELS_PER_SEED.csv) · [28行新模型指标](records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/reports/PER_SEED_RESULTS.csv)
- [核验记录](records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/VERIFICATION.json) · [训练及选模历史](records/jung_hang_cycle_v2/server_2b0ab5a_20261019/results/models/)
