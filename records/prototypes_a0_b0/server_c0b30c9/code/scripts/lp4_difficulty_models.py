"""任务难度对照：保留LP4/MD波形信息的45维谐波表示与小型MLP。

每相9维：直流项、1–4阶复系数的实部/虚部；五相共45维。
无幅值取绝对值、相位丢弃或数据驱动特征选择，且不读取标签/工况。
"""
import numpy as np
import torch
from torch import nn
from hust_arch_suite_models import FixedLowpass4
from v2_paifilter import RAW_POINTS, CNN_POINTS, deterministic_linear_resample

FEATURE_VERSION = "lp4_then_md_full_complex_k0_to4_45d_v1"
FEATURE_DIM = 45


def normalized_lp4(raw):
    """LP4后按原MD规则归一化，保持4096点以提取完整低阶复系数。

输入[B,5,4096]原电流；低电流相仍除100，其他相除中心化后的峰值。
公式与原MDNormalization一致，只暂缓最后的128点线性重采样。
    """
    if raw.ndim != 3 or raw.shape[1:] != (5, RAW_POINTS):
        raise ValueError("LP4特征要求[B,5,4096]电流")
    shaped = FixedLowpass4()(raw)
    centered = shaped - (shaped.amax(-1, keepdim=True) + shaped.amin(-1, keepdim=True)) * .5
    amplitude = centered.abs().amax(-1, keepdim=True)
    divisor = torch.where(centered.abs().mean(-1, keepdim=True) < .2,
                          torch.full_like(amplitude, 100.), amplitude.clamp_min(1e-8))
    return centered / divisor


def tensor_features(raw):
    """返回[B,45]，按相依次排列DC、Re1、Im1、…、Re4、Im4。"""
    spectrum = torch.fft.rfft(normalized_lp4(raw), n=RAW_POINTS, norm="forward")
    pairs = torch.view_as_real(spectrum[..., 1:5]).flatten(-2)
    return torch.cat((spectrum[..., :1].real, pairs), dim=-1).flatten(1)


def reconstruct_input128(features):
    """由45维重建原网络的5×128点输入，用来核验没有丢失LP4有效信息。

MD只施加每相的常数平移/缩放，仍是0–4阶带限信号。浮点FFT带来的
高阶舍入残差允许数值误差，不能误称重新采集或增加了原信息。
    """
    values = features.reshape(-1, 5, 9)
    spectrum = torch.zeros((len(values), 5, RAW_POINTS // 2 + 1),
                           dtype=torch.complex64, device=features.device)
    spectrum[..., 0] = values[..., 0]
    spectrum[..., 1:5] = torch.view_as_complex(values[..., 1:].reshape(-1, 5, 4, 2).contiguous())
    reconstructed = torch.fft.irfft(spectrum, n=RAW_POINTS, norm="forward")
    return deterministic_linear_resample(reconstructed, CNN_POINTS)


@torch.no_grad()
def extract_features(raw, device, batch=128, verify_reconstruction=False):
    """分批提取float32特征；可同时核对重建输入与原MDNormalization。

返回特征矩阵和最大重建绝对误差。特征缩放器另外只在训练数据上拟合。
    """
    from v2_paifilter import MDNormalization
    blocks, error = [], 0.
    for start in range(0, len(raw), batch):
        values = torch.from_numpy(raw[start:start + batch]).to(device)
        features = tensor_features(values)
        if verify_reconstruction:
            expected = MDNormalization()(FixedLowpass4()(values))
            restored = reconstruct_input128(features)
            error = max(error, float((expected - restored).abs().max().cpu()))
            torch.testing.assert_close(restored, expected, rtol=2e-6, atol=2e-6)
        blocks.append(features.cpu().numpy())
    features = np.concatenate(blocks)
    if features.shape != (len(raw), FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError("低阶谐波特征维度或数值异常")
    return features, error


class SmallHarmonicMLP(nn.Module):
    """45→32→6，单隐藏层ReLU、1670参数，不用卷积、BN或残差。"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.ReLU(), nn.Linear(32, 6))

    def forward(self, features):
        return self.layers(features)
