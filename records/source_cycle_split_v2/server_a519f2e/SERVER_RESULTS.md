# 全63文件仿真周期划分V2：服务器准备记录

## 范围与状态

- 日期：2026-10-15；协议：`hust-source-cycle-split-v2`。
- 代码提交：`a519f2e3a9a9df23d80a8dc82f3bace85944962f`；实现提交`0240df5`仅生成数据后遇到跨平台JSON键顺序校验差异，修正后的窗口和标签内容保持不变。
- 服务器：`wang@172.28.128.160`，复用原Python 3.12.3 / PyTorch 2.14.1+cu130环境，没有创建环境或安装依赖。
- 数据目录：`/home/wang/projects/会议论文/hust/datasets/source_cycle_split_v2`。
- 代码目录：`/home/wang/projects/会议论文/hust/experiments/source_cycle_split_v2/a519f2e/code`；其中`outputs/source_cycle_split_v2`链接到上述数据目录。
- 没有启动模型训练、生成权重或评估实测波形；旧数据、旧模型和旧结果保留。

## 划分结果

| 集合 | 原始文件 | 工况 | 周期数 | 健康0 | 故障1 | 故障2 | 故障3 | 故障4 | 故障5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 63 | 13 | 962 | 146 | 166 | 166 | 166 | 157 | 161 |
| Val | 63 | 13 | 221 | 38 | 37 | 37 | 37 | 36 | 36 |
| Gap，不参与训练/验证 | 63 | 13 | 78 | 15 | 13 | 13 | 13 | 12 | 12 |

- 使用全部63个MAT，不再为新协议单列仿真Source-Test；原174周期的旧文件仍保留在历史版本中。
- 59个标签来自作者Excel，4个按用户授权从命名补录，来源分别标记；400转/负载0只有故障1、2、3。
- 原始健康电流片段去重形成15组，保留确定来源；63个故障片段全部纳入。各片段按时间分块，先扣除1个隔离周期，再按约80/20划分Train/Val；本数据实际使用比例962/1183与221/1183。
- 两侧覆盖相同13组实际工况及所有存在的工况/故障组合，文件编号允许相同，时间范围和原始插值支撑点不相交。
- 最小间隔18.957毫秒，跨Train/Val的完整五相4096点float32样本完全重复数为0；同一记录内稳态周期的统计相关性仍然存在。
- 完整周期切分、4096点网格、模型算法及噪声档位保持原值。训练仍按整记录RMS加10–40 dB高斯噪声；验证为Clean与0/5/10/20/30 dB固定噪声，按需计算。
- 实测Dev/Holdout编号及窗口边界不变，本次没有读取实测波形。没有新划分下的模型成绩。

## 验证结果

- 本地75项测试通过、3项CUDA测试跳过；服务器78项全部通过。
- 本地与服务器8份JSON元数据逐字节相同；新划分SHA为`4a0c76fd8a9c9ff6dbf40c7413a8ec50cbf71694b6dcf1f324e06a97337eed3d`。
- 通用读取器返回Train `[962,5,4096]`、Val `[221,5,4096]`；五档固定验证噪声重复计算一致。
- 五档验证张量、种子42第1轮训练噪声张量在本地和服务器的SHA全部一致；该噪声检查不包含模型训练。
- 服务器原59个输入SHA保持不变，4个补录文件仅补传到本次独立输入目录。
- A1/B1当前CLI默认新数据和独立输出目录；显式`--legacy-source-split`保留旧数据复现。测试覆盖了默认路由、旧模式选择及禁止混合不同划分汇总。
- 初次Windows/Linux JSON语义完全一致，仅键的插入顺序不同。固定键排序及LF换行后，跨平台字节检查通过；初次生成目录在两端均保留，未覆盖。

## 文件

- [完整划分规则和命令](../../../docs/SOURCE_CYCLE_SPLIT_V2.md)
- [数据清单与指纹](metadata/SOURCE_SPLIT.json) · [63文件标签及来源](metadata/SOURCE_LABELS.json)
- [训练窗口](metadata/source_train_windows.json) · [验证窗口](metadata/source_val_windows.json) · [隔离窗口](metadata/source_gap_windows.json)
- [健康重复分组](metadata/HEALTHY_DUPLICATE_FAMILIES.json) · [逐工况计数](CONDITION_COUNTS.csv)
- [生成核查](metadata/VERIFICATION.json) · [服务器核验](SERVER_VERIFICATION.json) · [两端对账](LOCAL_SERVER_RECONCILIATION.json)
- [跨平台键顺序排查](INITIAL_ORDERING_DIAGNOSIS.json) · [实际测试日志](tests.log)

发布记录只包含代码、配置和元数据，不包含原始MAT、模型权重或环境。
