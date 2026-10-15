# 六类五相电流波形

共12张图：六个类别各有归一化前电流、128点预处理输入两张。每图三列为仿真、常规实测、10 dB实测，五行为A–E相。

每图使用1条仿真记录和1条实测基础记录。常规/10 dB两列对应同一实测记录与周期；三列不表示三个独立重复。
归一化前图使用实际电流单位及窗口起点之后的毫秒时间；实测同时间的测量已按实验规则取平均。预处理图直接读取本轮模型输入，横轴0–127。未按模型成绩筛选示例。

选择规则：目标Dev每类取最小记录编号；源Train按转速差、负载差、编号依次对应；每条取该类别窗口序列的中间周期。

图像为PNG 600 dpi，另有SVG/PDF；物理尺寸183×170 mm。文件与窗口索引、编号、哈希见[来源清单](../figures/waveforms/FIGURE_SOURCE_MANIFEST.json)。

| 类别 | 记录图 | 预处理图 |
|---|---|---|
| Healthy | [PNG](../figures/waveforms/00_healthy_before_normalization.png) | [PNG](../figures/waveforms/00_healthy_preprocessed_128.png) |
| OUS(A) | [PNG](../figures/waveforms/01_ous_a_before_normalization.png) | [PNG](../figures/waveforms/01_ous_a_preprocessed_128.png) |
| OLS(A) | [PNG](../figures/waveforms/02_ols_a_before_normalization.png) | [PNG](../figures/waveforms/02_ols_a_preprocessed_128.png) |
| OP(A) | [PNG](../figures/waveforms/03_op_a_before_normalization.png) | [PNG](../figures/waveforms/03_op_a_preprocessed_128.png) |
| OUS(A)&OUS(B) | [PNG](../figures/waveforms/04_ous_a_ous_b_before_normalization.png) | [PNG](../figures/waveforms/04_ous_a_ous_b_preprocessed_128.png) |
| OLS(A)&OP(C) | [PNG](../figures/waveforms/05_ols_a_op_c_before_normalization.png) | [PNG](../figures/waveforms/05_ols_a_op_c_preprocessed_128.png) |

## 0：Healthy

仿真记录`S-600-9-1-O`，实测记录`0081`。仿真周期0.252683–0.283220s，实测周期0.222491–0.283694s。

![Healthy归一化前电流](../figures/waveforms/00_healthy_before_normalization.png)

![Healthy128点预处理输入](../figures/waveforms/00_healthy_preprocessed_128.png)

## 1：OUS(A)

仿真记录`S-600-9-1-O`，实测记录`0081`。仿真周期0.744291–0.774827s，实测周期0.744739–0.805834s。

![OUS(A)归一化前电流](../figures/waveforms/01_ous_a_before_normalization.png)

![OUS(A)128点预处理输入](../figures/waveforms/01_ous_a_preprocessed_128.png)

## 2：OLS(A)

仿真记录`S-600-9-2-O`，实测记录`0085`。仿真周期0.744291–0.774827s，实测周期0.764843–0.797210s。

![OLS(A)归一化前电流](../figures/waveforms/02_ols_a_before_normalization.png)

![OLS(A)128点预处理输入](../figures/waveforms/02_ols_a_preprocessed_128.png)

## 3：OP(A)

仿真记录`S-600-9-3-O`，实测记录`0089`。仿真周期0.744288–0.774824s，实测周期0.777405–0.810878s。

![OP(A)归一化前电流](../figures/waveforms/03_op_a_before_normalization.png)

![OP(A)128点预处理输入](../figures/waveforms/03_op_a_preprocessed_128.png)

## 4：OUS(A)&OUS(B)

仿真记录`S-1000-6-4-O`，实测记录`4-1000-10`。仿真周期0.753194–0.772667s，实测周期0.752918–0.772374s。

![OUS(A)&OUS(B)归一化前电流](../figures/waveforms/04_ous_a_ous_b_before_normalization.png)

![OUS(A)&OUS(B)128点预处理输入](../figures/waveforms/04_ous_a_ous_b_preprocessed_128.png)

## 5：OLS(A)&OP(C)

仿真记录`S-1000-6-5-O`，实测记录`5-1000-7`。仿真周期0.750795–0.770087s，实测周期0.755781–0.777170s。

![OLS(A)&OP(C)归一化前电流](../figures/waveforms/05_ols_a_op_c_before_normalization.png)

![OLS(A)&OP(C)128点预处理输入](../figures/waveforms/05_ols_a_op_c_preprocessed_128.png)

## 绘图代码

[plot_waveforms.py](../scripts/plot_waveforms.py)使用Python/Matplotlib，输入为外部原数据、分析缓存和本轮已生成数组。`--dataset-repo`指定外部原始数据目录，`--project`指定工作目录；`--qa-helper-dir`为可选的图区几何核验工具目录。

本次绘图使用Python3.12.13、NumPy2.5.3、SciPy1.18.1、Matplotlib3.11.2。绘图依赖装在本地独立缓存目录，服务器训练环境未修改。

12张PDF逐张进行了文字字号、图区对齐和文字/线条重叠核对，结果见[QA_SUMMARY.json](../figures/waveforms/QA_SUMMARY.json)。PNG用于仓库页面展示，SVG/PDF为矢量文件。
