"""固定α=0.5的谐波融合对照：没有频谱描述符、门控网络或滤波参数。

输入为未归一化的完整电周期[B,5,4096]；输出保留0–3阶，第四阶乘0.5，
第五阶及以上置零。等价于0.5×LP3 + 0.5×LP4；仍按原流程MD归一化、
重采样128点，再由原25,620参数CNN分类。固定的是滤波器，CNN正常训练。
"""
import torch
from torch import nn
from v2_paifilter import RAW_POINTS, PHASES, OriginalCNN, MDNormalization


class FixedHalfFusion(nn.Module):
    """五相共享的实数频域掩码，保留各阶复数相位，不计算描述符。"""

    def forward(self, raw):
        """输入/输出均为[B,5,4096]，固定掩码不参与优化。"""
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError("固定融合需要[B,5,4096]完整周期")
        if not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
            raise ValueError("输入必须是有限的浮点电流")
        spectrum = torch.fft.rfft(raw, n=RAW_POINTS, dim=-1, norm="ortho")
        # 与V5相同的掩码形状与FFT算子；唯一差别是第四阶始终为0.5。
        mask = raw.new_zeros((raw.shape[0], RAW_POINTS // 2 + 1))
        mask[:, :4] = 1.
        mask[:, 4] = .5
        return torch.fft.irfft(spectrum * mask.unsqueeze(1), n=RAW_POINTS, dim=-1, norm="ortho")


class FixedHalfCNN(nn.Module):
    """原CNN先初始化，保证每个种子与V5初始CNN逐参数一致。"""

    def __init__(self):
        super().__init__()
        self.backbone = OriginalCNN()
        self.shaping = FixedHalfFusion()
        self.normalize = MDNormalization()

    def forward(self, raw):
        """滤波在MD归一化之前进行，返回六类分类logits。"""
        return self.backbone(self.normalize(self.shaping(raw)))
