"""V6：按原文重写的 Jung / Hang 输入构造与噪声增强。

本文件只做"从原始五相电流到模型输入"以及"按原文加噪"两件事，
网络结构直接复用 v3 的 JungModifiedCNN / HangWCNN（原文未公开逐层维度，
结构本身属于依据图示的重建，V6 不改动）。

V5 → V6 的六处修正，每处都对应原文的具体句子或可复现的数据事实：

1. Jung 输入窗口。
   原文 Sec. III-A："phase current data with a two-cycle length is sliced with a
   one-cycle size window"，即 2 个电周期是**录制长度**，喂给 CNN 的是**1 个
   电周期的窗口**；Sec. III-B 另有独立旁证："200 data points as input to
   represent a 50 Hz frequency at a 10 kHz sampling rate"。V5 误把录制长度
   当成窗口长度（2 周期 × 256 点），V6 改回 1 周期 × 200 点。

2. Hang 归一化。
   原文式 (5) |Is| = (3/2)Im 中的 Im 是该工况的定子电流幅值，是**标量常数**；
   式 (6) ain=Is/|Is| 有只含常数系数的闭合形式；原文还说健康与故障轨迹
   "have the same fixed radius"，故障形态含"straight lines"。逐采样时刻
   除以自身模长会把所有点钉在单位圆上，直线退化为两个点，与原文矛盾。
   V6 恢复为按标量归一化。

3. Jung 的 THD 谐波阶数。
   原文式 (1) 只统计 2、3、4 次谐波，V5 累加到 30 次，会高估原生 THD。

4. Jung 的噪声阻断范围。
   原文 Sec. IV-B 只对 Sj2/Sj3（半周期完全阻断）加限幅，Sj1/Sj4（幅值下降）
   不加。五相对应关系：OLS/OP 属于"相电流被完全阻断"，OUS 属于"幅值下降"。

5. 原生 THD 的 FFT 泄漏。
   矩形窗对非整数周期信号的旁瓣按 1/Δ 衰减，基波泄漏会把 2/3/4 次谐波
   所在谱点抬到约 2% 的假底噪。V5 因而在 3% 档只有 63.5% 的通道真正注噪。
   V6 先把参考段裁到整数个电周期，谐波才落在可测的谱点上。

6. Hang 的输入平面。
   原文对象是三相电机，Clarke 平面即完整的电流矢量空间；五相电机无中线
   时有四个独立自由度，对应基波与三次两个 VSD 平面。实测本数据集单相开路
   后基波轨迹仍为半径 1 的圆（各类平均图 L1 距离 0.003–0.011），故障信息
   落在三次平面（0.018–0.114）。因此 V6 使用双平面。
"""
from __future__ import annotations

import numpy as np

from jung_hang_models_v3 import (
    JungModifiedCNN, HangWCNN, five_phase_vsd, _draw_polyline,
    _interpolate_current_window,
)

NUM_CLASSES = 6

# 五相绕组空间相位（度）。属于模型假设，须用实际数据的相序核验后才能定稿。
FIVE_PHASE_ANGLES = (0.0, 72.0, 144.0, 216.0, 288.0)

# 各故障类中"相电流被完全阻断"的相。
# 依据：OUS(1)=开路开关，对应原文 Sj1/Sj4，只造成幅值下降，不限幅；
#       OLS(2)/OP(3)=开路桥臂/开路相，对应原文 Sj2/Sj3，半个周期无电流，限幅；
#       4=OUS(A)&OUS(B) 同为开关类，不限幅；5=OLS(A)&OP(C)，限 A、C 两相。
BLOCKED_PHASES = {
    0: (),
    1: (),
    2: (0,),
    3: (0,),
    4: (),
    5: (0, 2),
}


def record_reference(time, currents, reference_s=(0.10, 0.45),
                     phase_angles_deg=FIVE_PHASE_ANGLES):
    """估计一条记录共用的直流偏置与健康矢量模长（标量）。

    参数：
        time：记录时间轴，严格递增。
        currents：[T,5] 原始五相电流。
        reference_s：故障前健康区间，默认 (0.10, 0.45) 秒，与源数据划分一致。
        phase_angles_deg：五相绕组空间相位，必须与构造轨迹时使用的一致，
            否则尺度与轨迹不在同一坐标系下。

    返回：
        (dc, scale, used_reference)
        dc：[5] 逐相直流偏置，取健康段均值。
        scale：健康段 |I_alpha_beta| 的中位数，即原文式 (5) 的 Im
               （五相 0.4 幅值不变变换下健康矢量模长恒等于相电流幅值）。
        used_reference：本次是否真的用上了健康参考段。

    设计原因：
        原文的归一化只用来消除**负载**（电流幅值）差异，同一工况内是常数。
        用同一记录的健康段估计这个常数，既符合原文"For the healthy PMSM"的
        表述，也不会像"取窗口自身最大值"那样把故障造成的幅值下降重新拉伸回 1。
    """
    t = np.asarray(time, dtype=np.float64)
    x = np.asarray(currents, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 2 or x.shape[1] != 5 or len(t) != len(x):
        raise ValueError("expected aligned [T] time and [T,5] currents")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing")

    lo, hi = float(reference_s[0]), float(reference_s[1])
    selected = (t >= lo) & (t <= hi)
    used_reference = bool(selected.sum() >= 256)
    if not used_reference:
        # 记录过短或没有故障前健康段时退回整条记录，保证仍能给出标量尺度。
        selected = np.ones(len(t), dtype=bool)

    dc = x[selected].mean(axis=0)
    centered = x - dc
    components = five_phase_vsd(centered[selected], tuple(phase_angles_deg))
    magnitude = np.hypot(components[0], components[1])
    scale = float(np.median(magnitude))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("degenerate healthy current-vector modulus")
    return dc, scale, used_reference


def jung_window(time, currents, start, end, dc, points=200):
    """原文式：单电周期窗口，五相共用同一幅值尺度缩放到 [-1,1]。

    参数：
        dc：record_reference 给出的逐相直流偏置，在整条记录上去掉。
        points：每周期采样点数，默认 200，与原文"200 data points ... at
                10 kHz / 50 Hz"一致。

    返回：
        [5, points] float32。

    五相共用一个尺度而不是逐相各自归一化，是为了保留相间幅值比例——
    开路故障的判别信息主要就在"哪一相幅值掉了多少"，逐相归一化会抹掉它。
    去掉的是记录级直流偏置（传感器零点），不是窗口内均值；后者会把开路
    造成的波形单向缺失一并抵掉。
    """
    _, segment = _interpolate_current_window(
        time, np.asarray(currents, dtype=np.float64) - np.asarray(dc, dtype=np.float64),
        start, end, points,
    )
    scale = float(np.max(np.abs(segment)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros((5, points), dtype=np.float32)
    return (segment / scale).astype(np.float32)


def hang_trajectory(time, currents, start, end, dc, scale, *,
                    points=128, size=40, phase_angles_deg=FIVE_PHASE_ANGLES,
                    planes="dual"):
    """原文式 (5)(6)：把单周期电流矢量轨迹栅格化为轨迹图。

    参数：
        dc / scale：record_reference 给出的记录级直流偏置与健康矢量模长。
                    两者都是**整条记录共用的常数**，不是逐采样时刻量。
        points：轨迹重采样点数；40×40 栅格周长约 125 像素，128 点已足够。
        size：栅格边长，40 使两次 Haar-LL 池化后恰好落到原文 C7 的 10×10。
        planes："dual" 输出基波与三次两个平面（默认）；"fundamental" 只用基波平面。

    返回：
        [2, size, size] float32 的二值轨迹图（planes="fundamental" 时第 1 通道全零）。

    归一化形式与原文一致：Isn = (I_alpha + j I_beta) / Im。
    健康周期归一化后是半径 1 的整圆；单相阻断得到半圆/圆弧；
    两相阻断可以出现直线段。逐采样时刻除以自身模长则只会得到单位圆上的点，
    直线退化为两点——这正是 V5 失效的直接原因。

    为什么默认用双平面：
        原文对象是三相电机，三个相电流只有两个独立自由度，Clarke 平面就是
        完整的电流矢量空间。五相电机无中线时有四个独立自由度，对应基波与
        三次两个 VSD 平面，只用基波平面等于丢掉一半状态量。
        实测本数据集：单相开路后剩余四相仍能维持基波旋转磁动势，基波轨迹
        与健康几乎完全重合（各类平均图 L1 距离 0.003–0.011），故障信息主要
        落在三次平面（L1 距离 0.018–0.114）。因此双平面是五相任务下对原文
        "电流矢量轨迹"的必要推广，不是可选项。

    坐标尺度：
        两个平面共用同一个尺度，取 max(健康矢量模长, 本窗两平面的最大模长)。
        前一项与原文的常数 Im 一致；后一项只在三次平面幅值超过健康基波幅值
        时生效（实测 4/5 类会出现，最大 2.64 倍），作用是保证轨迹不被栅格边界
        截断。对健康与 1/2/3 类，这一项几乎不改变尺度（比值约 1.02）。
    """
    _, segment = _interpolate_current_window(
        time, np.asarray(currents, dtype=np.float64) - np.asarray(dc, dtype=np.float64),
        start, end, points,
    )
    components = five_phase_vsd(segment.T, tuple(phase_angles_deg))
    pairs = [(components[0], components[1])]
    if planes == "dual":
        pairs.append((components[2], components[3]))
    elif planes != "fundamental":
        raise ValueError("planes must be 'dual' or 'fundamental'")

    peak = max(float(np.max(np.hypot(alpha, beta))) for alpha, beta in pairs)
    render_scale = max(float(scale), peak, 1e-12)

    images = []
    for alpha, beta in pairs:
        x = alpha / render_scale
        y = beta / render_scale
        # 只跳过物理上不存在的零矢量（方向无定义），不设低电流比例阈值：
        # 阈值会人为丢弃故障造成的真实低幅值区段，降低类间差异。
        valid = np.hypot(x, y) > 1e-6
        images.append(_draw_polyline(x, y, size, valid=valid)
                      if valid.any() else np.zeros((size, size), dtype=np.float32))
    return np.stack(images).astype(np.float32)


def fundamental_thd(time, currents, omega, dc, *,
                    reference_s=(0.10, 0.45), harmonics=(2, 3, 4)):
    """原文式 (1)：用 2、3、4 次谐波估计基波 RMS 与原生 THD。

    参数：
        omega：瞬时角速度，用于给出标称基波频率 f0。
        dc：记录级直流偏置，先扣除再变换。
        reference_s：THD 在故障前健康段上估计。
        harmonics：原文式 (1) 只含 2、3、4 次，默认与原文一致。

    返回：
        (I1, thd)，两者都是长度 5 的数组，逐相给出基波 RMS 与 THD。

    为什么在健康段上估计：
        式 (1) 描述的是仿真电流本身固有的谐波水平（论文用开关频率在 1%~9%
        之间改变它）。开路故障会在故障段制造远超传感器噪声的阶跃畸变，
        若直接在故障段统计，会把"故障特征"误当成"原生 THD"，使绝大多数
        故障记录算出的许用噪声为零，与原文的增强意图相反。

    为什么必须先把参考段裁到整数个电周期：
        矩形窗对非整数周期信号的旁瓣按 1/Δ 衰减，第 12 个频点处的泄漏约为
        基波的 1/(π·12) ≈ 2.6%。若不裁剪，2/3/4 次谐波所在的谱点会被基波
        泄漏灌满，估算出的原生 THD 会被凭空抬到 2%~3% 的泄漏底噪上，
        于是目标 THD+N=3% 时几乎不注噪（V5 预检中 3% 档有 36.5% 的
        记录×相通道完全没有注入，就是这一效应）。裁剪到整数周期后谐波
        正好落在整数频点上，泄漏可忽略。
    """
    t = np.asarray(time, dtype=np.float64)
    x = np.asarray(currents, dtype=np.float64) - np.asarray(dc, dtype=np.float64)
    w = np.asarray(omega, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 5 or len(t) != len(x) or w.shape != t.shape:
        raise ValueError("expected aligned time / [T,5] currents / omega")

    lo, hi = float(reference_s[0]), float(reference_s[1])
    selected = (t >= lo) & (t <= hi)
    if selected.sum() >= 256:
        x, w, t = x[selected], w[selected], t[selected]

    dt = float(np.median(np.diff(t)))
    f0 = float(np.median(np.abs(w))) / (2.0 * np.pi)
    if not np.isfinite(dt) or dt <= 0 or not 0 < f0 < 0.5 / dt:
        raise ValueError("invalid sampling interval or fundamental frequency")

    # 裁到整数个电周期：周期信号在整数周期长度上严格周期，谐波恰好落在频点上。
    # 先用实际样本数能容纳的最大整周期数封顶，再换算回样本数，避免浮点边界
    # 导致裁出的长度超过数组本身。
    cycles = int(np.floor((len(t) - 1) * dt * f0))
    if cycles >= 4:
        keep = min(int(round(cycles / (f0 * dt))) + 1, len(t))
        if keep >= 4:
            x, w, t = x[:keep], w[:keep], t[:keep]

    frequency = np.fft.rfftfreq(len(t), dt)
    spectrum = np.fft.rfft(x, axis=0)
    near = np.flatnonzero((frequency >= 0.8 * f0) & (frequency <= 1.2 * f0))
    if not len(near):
        raise ValueError("no measured fundamental bin")
    # 取标称频率附近能量最强的谱线，减小转速估计误差带来的频谱泄漏。
    peak = near[np.argmax(np.abs(spectrum[near]).sum(axis=1))]
    f1 = float(frequency[peak])
    fundamental = np.sqrt(2.0) * np.abs(spectrum[peak]) / len(t)

    power = np.zeros(5, dtype=np.float64)
    for order in harmonics:
        if order * f1 >= frequency[-1]:
            continue
        index = int(np.argmin(np.abs(frequency - order * f1)))
        power += (np.sqrt(2.0) * np.abs(spectrum[index]) / len(t)) ** 2
    thd = np.sqrt(power) / np.maximum(fundamental, 1e-12)
    return fundamental, thd


def blocked_phase_mask(time, currents, omega, metadata, dc, quiet_ratio=0.03):
    """原文 Sec. IV-B 的限幅：只在相电流被阻断的区段屏蔽噪声。

    参数：
        dc：record_reference 给出的逐相直流偏置，必须先扣除。
            不扣直流时传感器零漂会直接抬高"被阻断"区段的局部 RMS，
            使限幅阈值失效（实测 2.5% 幅值的零漂即可让限幅器完全不起作用）。
        metadata：本记录窗口列表，每行含 label / start / end。
        quiet_ratio：局部 RMS 低于该相 95 分位的比例即视为"被阻断"。

    返回：
        与 currents 同形状的布尔掩码，True 表示该采样点允许加噪。

    原文对 Sj2/Sj3 用限幅器阻断半周期噪声，对 Sj1/Sj4 不加限幅；这里按
    BLOCKED_PHASES 把同一区分落到五相故障类上。这是拓扑不等价的任务适配，
    不等同于原 NPC 开关态规则。
    """
    t = np.asarray(time, dtype=np.float64)
    # 限幅器判断的是"电流有没有流动"，直流偏置不是电流，必须先去掉。
    x = np.asarray(currents, dtype=np.float64) - np.asarray(dc, dtype=np.float64)
    mask = np.ones_like(x, dtype=bool)
    dt = float(np.median(np.diff(t)))
    f0 = float(np.median(np.abs(np.asarray(omega, dtype=np.float64)))) / (2.0 * np.pi)
    width = max(3, int(round(0.03 / max(f0 * dt, 1e-9))))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float64) / width

    for row in metadata:
        for phase in BLOCKED_PHASES.get(int(row["label"]), ()):
            inside = np.flatnonzero((t >= row["start"]) & (t < row["end"]))
            if len(inside) < 3:
                continue
            local = np.sqrt(np.maximum(
                np.convolve(x[inside, phase] ** 2, kernel, mode="same"), 0.0))
            reference = float(np.percentile(local, 95))
            if reference <= 1e-10:
                mask[inside, phase] = False
            else:
                mask[inside, phase] = local > quiet_ratio * reference
    return mask


def jung_thdn_augment(time, currents, omega, target_percent, rng, *, dc,
                      metadata=(), limiter=True, quiet_ratio=0.03):
    """原文式 (5)：按目标 THD+N 注入白噪声。

    sigma = I1 * sqrt(max((THD+N)^2 - THD^2, 0))，逐相计算。
    目标值不高于原生 THD 时 sigma=0，即该相不加噪——这是式 (5) 的直接结果，
    不是额外的兜底逻辑。

    限幅屏蔽后**不再把噪声重新标定回目标 RMS**：原文的限幅就是"阻断噪声"，
    被阻断区段的噪声能量本就不应补到有效区段上。实际达到的 THD+N 会低于目标，
    这一点由 preflight 的覆盖率统计如实记录。
    """
    target = float(target_percent)
    if not 0 <= target <= 100:
        raise ValueError("THD+N percent must be within [0,100]")
    clean = np.asarray(currents, dtype=np.float64)
    fundamental, native_thd = fundamental_thd(time, clean, omega, dc)
    desired = target / 100.0
    sigma = fundamental * np.sqrt(np.maximum(desired ** 2 - native_thd ** 2, 0.0))

    noise = rng.standard_normal(clean.shape)
    if limiter:
        noise *= blocked_phase_mask(time, clean, omega, metadata, dc, quiet_ratio)
    noisy = clean + noise * sigma[None, :]
    if not np.isfinite(noisy).all():
        raise FloatingPointError("non-finite THD+N augmented waveform")
    return noisy


def add_awgn(currents, snr_db, rng):
    """原文式 (17)：逐相白噪声，按整记录信号功率标定到给定 SNR。

    SNR = 10*log10(Psignal / Pnoise) => Pnoise = Psignal * 10^(-SNR/10)。
    """
    if not -10.0 <= float(snr_db) <= 80.0:
        raise ValueError("snr_db outside supported range")
    signal = np.asarray(currents, dtype=np.float64)
    ratio = 10.0 ** (-float(snr_db) / 10.0)
    noise = rng.standard_normal(signal.shape)
    for phase in range(signal.shape[1]):
        power = float(np.mean(signal[:, phase] ** 2))
        if power <= 1e-18:
            noise[:, phase] = 0.0
            continue
        noise[:, phase] *= np.sqrt(power * ratio)
    return signal + noise
