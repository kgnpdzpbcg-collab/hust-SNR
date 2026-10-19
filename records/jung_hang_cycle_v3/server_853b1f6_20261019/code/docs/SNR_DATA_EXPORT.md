# HUST-SNR：仅导出冻结目标 Dev 波形（非训练脚本）

## 目的和边界

为独立分析 M1 在低 SNR 下的诊断错误，导出 **原始电流（未进行 128 点重采样、归一化、滤波）**。所使用的原始记录编号严格来自 `configs/splits_gate0_v1.json` 的 `target_dev_records`（应为 **88** 条），不能选择剩余的 **347** 条 Final-Holdout。

源域训练文件、正式权重、目标测试指标均不会被修改；不会进行模型训练或 Final-Holdout 推断。公开 GitHub 仓库**只存脚本，不提交导出的 `.npz`**（`outputs/` 已在 `.gitignore` 中）。

## 在原服务器运行

进入仓库的 `research/noise-consistency-v1` 分支；确保以下本地文件存在：

- `configs/splits_gate0_v1.json`
- `analysis/data_audit_v1/LABELS.json`（本地审计生成，gitignored）
- `outputs/gate0_v1/target_dev_clean_windows.json`（Gate 0 冻结周期清单，gitignored）
- `data/target_dev/clean/<record_id>.mat`
- 七档含噪 MAT（变量名 `I_noisy`），例如 `<SNR_ROOT>/SNR_10/0081.mat`。

```bash
python -m pytest -q tests/test_export_snr_data.py

python scripts/export_snr_data.py \
  --snr-root "/path/to/Five-phase-IM-AA-experimental-domain-different-SNR-data-set"
```

Windows PowerShell：

```powershell
python scripts/export_snr_data.py --snr-root "D:\\project\\hust\\Five-phase-IM-AA-experimental-domain-different-SNR-data-set"
```

上述路径为**示例**，请改成服务器原始数据真实位置。注意根目录下面应直接存在 `SNR_10/` 至 `SNR_40/`。

默认输出是 `<repo>/outputs/dev_snr_export/`：**7 个 `dev_snrXX.npz`＋一个 `dev_clean.npz`＋一个 `dev_reference.npz`＋ `dev_cycle_windows.json`＋ `manifest.json`**。如不用 Clean 导出可加 `--no-clean`。输出目录不得事先存在，防止覆盖。

## 文件结构

```text
outputs/dev_snr_export/
├── manifest.json
├── dev_reference.npz          # 88条记录的编号、时间、转速、电机故障标签、转速/转矩工况
├── dev_cycle_windows.json    # 原冻结窗口清单：record_id/start/end/label，保持 V1 排序
├── dev_clean.npz             # 可选；未归一化五相干净实测电流
├── dev_snr10.npz             # 10 dB 未归一化五相实测电流
├── dev_snr15.npz
├── dev_snr20.npz
├── dev_snr25.npz
├── dev_snr30.npz
├── dev_snr35.npz
└── dev_snr40.npz
```

`dev_snrXX.npz` 内 `currents` 维度是 `(88, 10011, 5)`，类型 `float32`；`record_indices` 为本文件每条记录在共享 `dev_reference.npz` 中的行号。`dev_reference.npz` 里的 `record_ids`、`time`、`omega`、`fault_label`、`speed`、`torque` 的排列完全一致，时间和角速度使用 `float64`。重复时间戳按照 **Gate 0/V1 的均值合并规则**处理。

**标签说明：**`fault_label` 是记录后半段的故障类别（1–5），记录的故障前健康区间是类别 `0`。真正每个模型输入窗口的标签以 `dev_cycle_windows.json` 为准，不能整条记录都贴故障标签。

每一个含噪 MAT 与 Clean MAT 先在原始 10031 个采样点上检查时间、转速逐元素一致，并按每相原始信号功率重算标称 SNR（允许误差 < `1e-6` dB）；校验通过后将电流转成 `float32`、打包并回读验证。输出 `manifest.json` 记录文件 SHA256，允许后来校验上传完整性。

## 文件太大时，自动分包

如果对话文件上传限制比较紧，推荐一次导出每文件不超过 20 条记录：

```bash
python scripts/export_snr_data.py \
  --snr-root "/path/to/SNR_parent" \
  --records-per-file 20 \
  --output outputs/dev_snr_export_parts
```

此时 `dev_snr10_part01.npz` 等会分成若干部分，每个部分保存**全局** `record_indices`，因此后续仍能准确对齐记录编号和 V1 预测。每个部分未压缩的电流阵列约 4.0 MB，压缩后大小取决于波形；共享时间/转速文件单独上传一次即可。

## 供分析端读取

```python
import numpy as np
from pathlib import Path

root = Path("outputs/dev_snr_export")
with np.load(root / "dev_reference.npz", allow_pickle=False) as ref:
    ids = ref["record_ids"]
    time = ref["time"]
    omega = ref["omega"]
with np.load(root / "dev_snr10.npz", allow_pickle=False) as d:
    idx = d["record_indices"]
    currents = d["currents"]
    print(ids[idx[0]], currents.shape, time[idx[0]].shape)
```

**科学边界：**这些 SNR 文件是同一批实测记录的不同受控噪声版本，**不是不同工厂或独立试验**。输出 `float32` 是轻微有损转换，仅用于波形特征和分类错误分析；如需验证微小噪声数值差异，以原服务器上的 `float64` MAT 为准。数据发布者的具体加噪机制未披露，脚本不推断其噪声分布。
