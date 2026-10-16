# HUST 经典方法基线：RBF-SVM / Random Forest / 1D-ResNet（V1）

## 研究用途与边界

新增三种标准故障诊断模型，与已有 A′、LP3/LP4、NA-HSF、Xu-Full 进行**整体方法效果对照**；不更改原有模型、权重和报告，不假定任一基线必然弱于本文方法。本实验是独立 `research/classical-baselines-v1` 分支与独立输出目录。尚未运行真实服务器训练时，不得填写真实准确率。

- 严格只使用同一 Source-Train（34 条 recording／738 个完整电周期）拟合模型；Source-Val（15 条／302 周期）用于选超参数或选 epoch。Target-Dev（88 条／2474 周期）**仅在训练和选择结束之后**用于推理及统计；最终 347 条 Holdout 不访问。
- 固定三种子 **42/43/44**；ResNet 使用三次独立初始化，RF 使用三组随机树，SVM 采用三组预先定义的源域噪声增强视图。不得把 SVM 三组相似结果解释成由随机权重初始化引起的波动。
- 训练噪声集合 10、15、20、25、30、35、40 dB；Source-Val 冻结 Clean＋0/5/10/20/30 dB。Target-Dev 为 Clean、10–40 dB 七档原始作者噪声文件及额外 0/5 dB 合成强噪声。
- 源域噪声生成重用 `v2_experiment.fresh_noisy_source()`（整 recording、每相 RMS 高斯噪声）。0/5 dB 的源验证沿用以前核验的固定 source MAT；目标 0/5 dB 是作者 10 dB 噪声残差放大所得，**并非真实测量的新 SNR 数据**。

## 模型规格

| Model | Input representation | Training | Selection |
|---|---|---|---|
| SVM (RBF) | 五相完整 4096 点电周期→每相 MD 归一化→每相 8 个时域统计＋前 8 阶谐波幅值＋前 4 阶谐波复系数实部和虚部，共 **120 维** | Source-Train Clean＋7 次冻结随机增强；`StandardScaler` 只在这 5904 个源域训练视图上 `fit`；`SVC(C∈{0.1,1,10}, gamma='scale')` | 使用五档 Source-Val，不在 Target-Dev 上调 C |
| Random Forest | 完全相同的 120 维特征 | 同样 5904 个源训练视图；300 trees，`min_samples_leaf∈{1,3}`，`max_features='sqrt'` | 五档 Source-Val 选叶子数，仅源域 |
| 1D-ResNet | 五相 4096 点→**同 V5** 的可微 MD 归一化／128 点采样→卷积残差网络；不做频域滤波 | Stem 5→32（7 点核），3 个残差阶段 32/64/128 通道、各 2 个 BasicBlock，下采样步长2，自适应全局平均池化＋线性六分类。50 epoch、batch32、Adam 0.001、WD 0；Clean/Noisy 配对平均 CE，噪声按 V5 每轮重新生成 | 与 V5 完全相同的五档 Source-Val checkpoint 选择规则 |

**公平性声明**：三者与现有 V5 使用相同的原始 source／target 记录及评价协议，但不能说三种训练算法拥有完全相同的优化步骤或数据视图总数。SVM/RF 因为是非增量的 sklearn 训练，用 7 个独立源域噪声视图构成固定训练集，且要仅用源域选超参数；ResNet 使用 50 轮逐 epoch 更新的源域配对噪声，与现有 V5 对齐。差异写进论文实验设置即可。

### 120 维特征定义

逐相 8 个统计：归一化均值、RMS、平均绝对值、标准差、峰峰值、峰值/RMS、偏度、峰度；频域取整数谐波阶次 `k=1...8` 的幅值，以及 `k=1...4` 的 FFT 复数系数实部、虚部。先在五相完整周期的原始长度（4096）上计算 FFT，保留五相的原始相序和相位，全部计算与标签无关。5 相 × 24 维 = 120 维。没有针对 Target-Dev 设计或筛选手工特征。

### Frozen 模型选择

对 SVM / RF 的候选超参数模型，首先确定 Source-Val Clean Macro-F1 最高值，候选模型须在其 1 个百分点以内，再最大化 Source-Val 的 0/5/10/20/30 dB 平均 Macro-F1；并列时按 Clean 和候选配置顺序确定。ResNet 则在其 50 个 epoch 上应用原 V5 同一规则。

### 审计输出

每个方法／种子的 `run.json` 记录源域选择条件、训练规模、最优超参或最佳 epoch。`best.joblib`／`best.pt` 存在服务器 `outputs/` 下，不随 GitHub 发布；不能加载外部来源的不可信 joblib。对十档 Target-Dev 保存真实标签和预测（每条件 2474 个周期）以及六类 Recall/F1、混淆矩阵、误报、漏报；`reports/PER_SEED_RESULTS.csv`、`PER_CLASS_PER_SEED.csv`、`SUMMARY.json` 汇总三种子成绩。不得通过 Target-Dev 结果选择更好随机种子或修改模型。禁止任何 Holdout 扫描。

## 实际服务器命令

环境中须已有 `torch`、`numpy`、`scipy`、`scikit-learn`、`joblib`、`pytest` 等依赖；**不对原服务器自行升级依赖**。在仓库根目录：

```bash
export SOURCE_STRONG_ROOT="/path/to/strong_data_root"
export TARGET_SNR_ROOT="/path/to/original_snr_parent"
export TARGET_STRONG_ROOT="/path/to/strong_data_root"
export PYTHON_BIN="/path/to/existing/venv/bin/python"  # optional
bash scripts/run_classical_baselines_server.sh
```

三种方法将各自执行 42/43/44 三种子训练、十条件测试和汇总。脚本拒绝覆盖已完成的工作；为重跑分配新的 `CLASSICAL_OUT`。`source_strong_root` 应包含 `generated/source/SNR_0/` 和 `generated/source/SNR_5/`，`strong_root` 应包含 `generated/target/SNR_0/5/`；原始目标噪声父目录应包含 `SNR_10` 至 `SNR_40`（或现有读取器认可的路径）。

如只测试某个模型，可以调用独立入口，例如：

```bash
python scripts/classical_baseline_experiment.py train \
  --methods resnet1d --seeds 42 43 44 \
  --source-strong-root "$SOURCE_STRONG_ROOT" --out outputs/classical_baselines_v1
```

## 论文解释边界

经典算法使用人工特征，ResNet 直接学习波形，Xu-Full 使用自编码器和 DML，V5 使用谐波融合。各模型的特征／主干不同，因此它们之间的差异是**整体方法比较**，不意味着 NA-HSF 单个模块产生了全部提升。模块必要性仍以原 A′ 家族的 LP3/LP4/Fixed-0.5/NA-HSF 消融为依据。三种子、所有条件的结果要完整报告；如果其他方法成绩更好，同样如实呈现。

## 服务器完成记录

2026-10-13，运行提交`4860c77`完成九组训练、90组十条件Dev评估及独立预测重放。正式数值、源选模、文件/类别/时间段索引见[服务器记录](../records/classical_baselines/server_4860c77/SERVER_RESULTS.md)。原始文件含故障前后不同状态，文件索引须按类别分行；分类始终采用已冻结的逐周期标签。本协议参数保持不变。
