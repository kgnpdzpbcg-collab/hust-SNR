# −10 dB数据生成记录

## Material Passport

- 类型：服务器数据生成及核验记录；客户端日期：2026-10-15。
- 来源：本目录GENERATION_MANIFEST.json、INDEPENDENT_VERIFICATION.json及GENERATION_COMMAND.json。
- 状态：499份MAT实际生成并独立核验通过；没有训练或模型推理。

## 文件与保存位置

- 新仿真：`/home/wang/projects/会议论文/hust/datasets/strong_snr_minus10_v1_20261015/generated/source/SNR_-10/`，59份。
- 新实测：`/home/wang/projects/会议论文/hust/datasets/strong_snr_minus10_v1_20261015/generated/target/SNR_-10/`，440份。
- 合计499份，210,172,642字节。MAT保存在服务器，本仓库仅保存代码、配置、索引和核验记录。
- 仿真原文件：`/home/wang/projects/会议论文/hust/data/source/`；实测干净文件与作者10 dB文件：`/home/wang/projects/会议论文/hust/datasets/strong_snr_v1_20261012/inputs/target_clean/`和`target_snr10/`。

## 生成规则

仿真仍按整条记录、每相电流的均方功率校准高斯噪声，目标SNR为−10 dB。原base_seed=20261012；验证随机流为`HUST-V1:val:-10:{编号}`，其他划分沿用原命名规则。随机数种子、每相实测SNR和文件SHA均逐文件保存。

实测逐编号配对计算：`新电流 = 干净电流 + (作者10 dB电流 − 干净电流) × 10`。只缩放作者已有噪声残差，没有向旧0/−5 dB文件叠加新随机噪声。各噪声档位共享同一噪声实现，不是独立采集。−10 dB表示每相整记录噪声功率为信号功率的10倍；不表示每个切出周期的局部SNR都等于−10 dB。

仿真矩阵仍为`I_We`、10000×6；仅前五列电流改变，末列转速保持。实测矩阵仍为`I_noisy`、10031×7；仅五相电流改变，时间列、转速列及原重复时间戳保持。

| 原划分 | 本轮生成文件数 |
|---|---:|
| 仿真训练 | 34 |
| 仿真验证 | 15 |
| 仿真保留 | 10 |
| 实测开发评估 | 88 |
| 实测最终保留 | 347 |
| 实测重复别名 | 5 |

本步骤生成全部440个实测编号；后续模型评估仅使用原88个Dev编号。生成最终保留集的派生波形不等于评估该保留集。

## 核验记录

- 499份MAT重新加载后与生成矩阵一致，SHA核对通过；最大整记录每相SNR误差5.151e-14 dB。
- 59份仿真均可从固定种子逐数值复现；15份源验证与原`calibrated_noise`逐数值一致。
- 440份实测均与作者10 dB残差×10逐数值一致；与已有0 dB残差×√10、−5 dB残差×10^0.25核对通过。
- 原939份输入、998份0/5 dB文件、499份−5 dB文件未变；4843个受保护文件SHA不变，其中171份模型权重。
- 12项生成测试通过。生成子进程耗时6.977秒；未安装依赖、未修改原训练或验证噪声配置。

[逐文件索引](GENERATED_FILE_INDEX.csv) · [完整生成清单](GENERATION_MANIFEST.json) · [独立核验](INDEPENDENT_VERIFICATION.json)
