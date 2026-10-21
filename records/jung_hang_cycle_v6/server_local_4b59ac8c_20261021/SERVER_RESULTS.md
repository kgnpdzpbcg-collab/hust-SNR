# 本地Jung/Hang V6：服务器实验记录

## Material Passport

- 执行日期：2026-10-21；模式：run；四组训练、28组七档Dev评价和回放已完成。
- 输入版本：用户本地V6，原始七文件快照SHA256：`4b59ac8c0a06741dd723f84b13ba0d1cfe58348f831274a9995c2afa4e314301`。
- 数据/公共模块底座提交：`a0ae214f36f69e873c0f23ba85c36fd304921f04`；该提交本身不是V6模型提交。
- [原始本地源码快照](local_original/)和[源码指纹](LOCAL_SOURCE_MANIFEST.json)保留。
- 两模型为论文机制指导下的五相任务适配，不是作者官方源码逐参数复现。

## 验证入口修复

- 原本地noise_records把Jung固定验证的dB数值传给了THD+N百分比增强函数，原0 dB验证实际未加噪。
- 唯一代码修复：THD+N仅在Jung训练（snr=None）执行；两个模型的固定验证都调用SNR-AWGN。
- 模型结构、训练噪声、优化器、配置及原十项模型测试保持本地原样；新增八项入口回归测试。
- [原问题复现](ORIGINAL_VALIDATION_BUG.json)；[最小补丁](VALIDATION_SNR_FIX.patch)；[入口回归测试](code/tests/test_jung_hang_cycle_v6_noise.py)。

## 记录级参考段及划分边界

- V6使用同记录0.10–0.45秒健康参考段估计逐相直流偏置与Hang标量尺度。Jung也使用该段估计的直流偏置。
- 训练和验证输入各自从整条源记录构造参考，参考区间覆盖了38个源验证健康窗口；训练预处理并非只使用Train时段。
- 实测测试从对应SNR的同一含噪记录估计参考，不读取其干净对应波形；参考可能包含当前健康测试窗口之后的采样。
- 这些是本地V6的现有设计，本轮未更改。当前结果记录为带记录级健康参考校准的版本实验，不标为仅当前窗口的严格独立预处理协议。
- [参考段审计](REFERENCE_PROTOCOL_AUDIT.json)；[协议说明](results/reports/PROTOCOL_NOTICE.json)。

## 数据与训练配置

- 原Source-Cycle V2清单不变：63记录、962 Train、221 Val、78 Gap；划分SHA256保持冻结值。
- 两模型均使用单电周期的全部962/221窗口；Jung输入5×200，Hang双VSD平面2×40×40。
- Jung：健康参考段整数周期FFT，原生THD仅统计2/3/4阶；训练THD+N=3%，仅标签2/3/5按BLOCKED_PHASES限幅。限幅后不再把噪声重新标定回目标RMS。
- Hang：基波和三次VSD双平面，共享标量尺度max(记录健康模长,当前窗两平面峰值)，训练15/20 dB AWGN。
- 种子42/44；Jung为AdaMod、30轮；Hang为Adam、50轮；lr=0.001、weight_decay=0、batch32。
- 两模型每批配对Clean/Noisy，CE各占0.5；每轮31次更新，共160轮、4960次更新。
- 修复后固定源验证为Clean与0/5/10/20/30 dB AWGN；两方法使用不同HUST-V6方法命名空间的随机噪声。
- 源选模依次使用Clean距峰值≤1个百分点、最大五噪声平均F1、Clean F1、最小五噪声平均CE及最早轮次。
- 实测使用原88个Dev编号、每档2474周期，5/4/0/−4/−5/−8/−10 dB文件不重新生成。
- 实测数据不用于权重拟合或checkpoint选择；上述参考预处理访问已单独披露。未读取347条Final-Holdout波形。
- 42/44种子延续用户在查看历史六种子结果后的指定；本轮未增加种子，也未运行PMSM。

## 源域预检

- 正式pytest共26项通过：V3公共网络8项、本地V6模型10项、验证/训练噪声入口回归8项。两模型真实源样本GPU梯度有限。
- 原生THD中位数0.05799%，最大0.10210%；3%档非零注噪覆盖率100%。
- Hang训练图不同图像605种，跨标签完全相同组0组；验证图不同165种，跨标签组0组。
- 健康参考段没有回退到整记录的源文件。相序证据复用同一源清单和同一VSD角度的已验证源波形拟合，最大偏差0.01107度，不表示硬件接线独立认证。
- [源预检](results/PREFLIGHT_PAPER_AUDIT.json)；[真实源记录验证SNR检查](DEPLOYMENT_MANIFEST.json)。

## 参数及源选模

| 模型 | 种子 | 参数量 | 最佳轮次 | Clean源F1（%） | 五噪声源F1（%） |
|---|---:|---:|---:|---:|---:|
| Jung V6 | 42 | 7110 | 22 | 100.00 | 90.02 |
| Jung V6 | 44 | 7110 | 29 | 100.00 | 89.52 |
| Hang V6 | 42 | 63330 | 37 | 100.00 | 86.89 |
| Hang V6 | 44 | 63330 | 2 | 100.00 | 90.87 |

## 种子42：Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jung V6 | 52.65 | 50.25 | 46.06 | 29.39 | 28.50 | 27.02 | 24.58 |
| Hang V6 | 39.87 | 34.84 | 15.26 | 10.50 | 10.14 | 9.35 | 8.52 |

## 种子44：Macro-F1（%）

| 模型 | 5 dB | 4 dB | 0 dB | −4 dB | −5 dB | −8 dB | −10 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jung V6 | 39.31 | 37.42 | 31.67 | 25.36 | 24.85 | 23.50 | 22.47 |
| Hang V6 | 37.26 | 35.47 | 14.36 | 12.67 | 12.52 | 12.11 | 11.26 |

## 核验、环境及文件

- 28组best实测预测、24组best源验证F1/CE与源选模回放均通过；推理不改变模型参数或BN。
- sklearn独立复算Macro-F1、Accuracy、六类F1/Recall和混淆矩阵；冻结元数据真值向量逐文件核对。
- 1745个历史输入、代码、权重与结果指纹不变；新best/last共8份留在服务器，归档只含SHA。
- GPU：NVIDIA GeForce RTX 5070 Ti；PyTorch `2.14.1+cu130`；复用解释器`/home/wang/projects/会议论文/analog-circuit-crossdomain/.venv/bin/python`，未新建环境。
- 并发训练墙钟112.532秒；测试37.078秒；核验41.802秒。
- 训练/评价/核验/汇总总墙钟192.432秒，不含部署、预检和整理。
- [28行逐种子指标](results/reports/PER_SEED_RESULTS.csv)；[168行逐类指标](results/reports/PER_CLASS_PER_SEED.csv)；[160轮历史](results/reports/TRAINING_HISTORY.csv)。
- [完整训练与选模记录](results/models/)；[28份预测](results/evaluations/)；[24组源验证回放](results/reports/SOURCE_VALIDATION_REPLAY.csv)；[核验状态](results/VERIFICATION.json)。
- [实际运行代码](code/)；[训练测试日志](logs/)；[部署清单](DEPLOYMENT_MANIFEST.json)；[导出核验](EXPORT_VERIFICATION.json)。
- [服务器数据路径启动包装器](ENTRY_BOOTSTRAP.py)只预载入既有冻结读取器，不修改V6算法；实际命令见[流程记录](PIPELINE_STATUS.json)。
- 上游汇总器附带十四模型表保持原样，但V6参考段信息访问与旧模型不同；没有据此做直接排名。
- 原始MAT、权重、环境、高维NPZ未包含在发布内容中。
