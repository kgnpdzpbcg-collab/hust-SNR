# 全63文件的仿真周期划分V2

## 范围与版本

- 用户于2026-10-15授权调整仿真Train/Val，并同意按文件名补录Excel遗漏的4个标签。
- 数据协议为`hust-source-cycle-split-v2`；配置为`configs/source_cycle_split_v2.json`。
- 原63个MAT全部纳入，共13组转速/负载组合；不再单独保留仿真Source-Test。
- 每个样本仍为一个完整电周期的五相电流，前端网格仍为每相4096点；模型结构、归一化、训练与验证噪声档位没有调整。
- 本次仅准备数据并验证读取，没有启动模型训练或实测评估。

## 标签来源

原Excel中的59个文件逐项与命名规则`S-转速-负载-故障编号-O`交叉核对；转速、负载、故障编号及名称全部一致。其余4个条目按相同命名规则补录，保留`filename_inferred_user_authorized`标记，不写成作者Excel原有标签。

| 补录文件 | 故障编号 | 故障名称 |
|---|---:|---|
| S-400-0-1-O.mat | 1 | OUS(A) |
| S-400-0-2-O.mat | 2 | OLS(A) |
| S-400-0-3-O.mat | 3 | OP(A) |
| S-1000-9-4-O.mat | 4 | OUS(A)&OUS(B) |

新增400转/负载0工况仅有故障1、2、3；没有人为补造该工况的故障4、5。

## 分块与去重

1. 沿用10kHz时间轴和角速度累计2π切完整周期。健康区间0.1–0.45秒；故障区间从0.5秒后跳过两个周期至0.98秒。
2. 对0.1–0.45秒健康片段的五相float64电流按原字节去重，形成15个唯一片段，每组保留字典序首个文件。全部63个故障片段保留。
3. 每个片段的周期按时间排序。先扣除1个边界隔离周期，验证数为剩余周期数的20%四舍五入、至少1个；早段Train、隔离周期Gap、末段Val。因此总比例为约81.3%/18.7%，不强制每个短片段精确80/20。
4. 78个片段各留1个Gap周期，不进入训练或验证。所有实际存在的工况/故障组合均同时出现在Train、Val。
5. 划分先于加噪，同一基础周期只属于一个集合。训练仍按整记录、每相RMS标定10/15/20/25/30/35/40 dB高斯噪声，每轮更新；验证0/5/10/20/30 dB使用原固定rid/SNR随机流，独立于训练噪声生成。验证噪声按需生成，原59文件缓存不必改写。

| 集合 | 涉及原文件 | 周期样本 | 健康0 | 故障1 | 故障2 | 故障3 | 故障4 | 故障5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 63 | 962 | 146 | 166 | 166 | 166 | 157 | 161 |
| Val | 63 | 221 | 38 | 37 | 37 | 37 | 36 | 36 |
| Gap，不使用 | 63 | 78 | 15 | 13 | 13 | 13 | 12 | 12 |

Train和Val的文件编号会相同，时间块与原始插值支撑点不相交；不能继续用旧协议的文件编号互斥断言。实际最小时间间隔为18.957毫秒，五相4096点float32周期的跨集合完全重复数为0。

同一记录的稳态周期可能高度相似，去重和时间间隔不代表独立采集，也不保证源验证不会满分。本协议验证用于同工况内的选模；真实域开发集和最终保留集边界保持原样。

## 数据准备与读取

在既有Python环境运行；源目录直接包含63个MAT，输出目录必须不存在：

```powershell
python scripts/source_cycle_split.py --source-root "原63个MAT目录" --labels-xlsx "Five-phase-IM-AA-simulation-domain.xlsx" --out outputs/source_cycle_split_v2
```

输出包含`raw_source/`私有输入副本、`SOURCE_LABELS.json`、输入SHA、健康去重分组、Train/Val/Gap窗口、`SOURCE_SPLIT.json`和`VERIFICATION.json`。数据目录位于被忽略的`outputs/`，原始MAT不提交Git。

通用接口`source_cycle_split.source_views(root)`返回与历史读取器相同的两组结构，包含原始记录、窗口、标签、4096点Clean和固定Val噪声；不依赖`738/302`硬编码。

## A1/B1训练入口的版本选择

当前A1/B1命令行默认读取新目录，默认输出到`outputs/hust_prototypes_a1_b1_source_cycles_v2`；以下命令仅作后续使用说明，本次没有执行：

```bash
python scripts/hust_a1_b1_experiment.py train --methods a1 b1 --source-split-root outputs/source_cycle_split_v2
```

模型训练参数保持原配置。权重、run和评估记录另存数据版本及窗口清单SHA，汇总时拒绝混入不同数据划分的结果。旧协议可显式复现，默认输出仍为原目录且拒绝覆盖已有权重：

```bash
python scripts/hust_a1_b1_experiment.py train --legacy-source-split --source-strong-root /path/to/old_strong_root
```

历史归档中的代码和全部原结果保持原字节；其他历史实验入口保留原协议复现用途，后续新训练须显式使用同一V2数据读取接口。

## 本次准备状态

服务器已生成并核查同版本数据，目录为`/home/wang/projects/会议论文/hust/datasets/source_cycle_split_v2`。本地与服务器8份JSON及五档验证张量指纹一致；服务器78项测试通过。本次没有模型训练或实测评价。

[完整准备记录、13组工况计数及核验](../records/source_cycle_split_v2/server_a519f2e/SERVER_RESULTS.md)。原始MAT、环境和模型权重不在发布记录中。

后续实验：2026-10-15已按此划分完成[七模型六种子比较](../records/source_cycle_benchmark/server_4728e9b/SERVER_RESULTS.md)。本页“未训练”指此前独立的数据准备步骤。
