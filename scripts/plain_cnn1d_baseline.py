"""A plain 1D CNN matched to the ResNet1D baseline, without residual shortcuts.

Independent comparator, NOT the proposed V5 architecture. Same input processing,
stem, six 2-convolution stages, channel widths, downsampling, GAP and classifier
as the existing ResNet1D. No skip connections or projection shortcuts.
"""
from __future__ import annotations
import torch
from torch import nn
from v2_paifilter import MDNormalization, RAW_POINTS, PHASES, CLASSES


class PlainConvBlock(nn.Module):
    """Two Conv1D-BN layers with an intermediate and final ReLU, no addition."""
    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FivePhasePlainCNN1D(nn.Module):
    """No-skip 1D CNN: [B,5,4096] -> MD [B,5,128] -> 128-d GAP -> 6 logits.

    Paired with FivePhaseResNet1D. Parameters differ because ResNet includes
    learned projection shortcuts at channel transitions; report this honestly.
    """
    def __init__(self):
        super().__init__()
        self.normalize = MDNormalization()
        self.stem = nn.Sequential(
            nn.Conv1d(PHASES, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            PlainConvBlock(32, 32), PlainConvBlock(32, 32),
            PlainConvBlock(32, 64, stride=2), PlainConvBlock(64, 64),
            PlainConvBlock(64, 128, stride=2), PlainConvBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, CLASSES)

    def forward(self, raw: torch.Tensor, return_features: bool = False):
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (PHASES, RAW_POINTS):
            raise ValueError(f"Expected [B,{PHASES},{RAW_POINTS}], got {tuple(raw.shape)}")
        features = self.pool(self.layers(self.stem(self.normalize(raw)))).flatten(1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def smoke() -> None:
    torch.manual_seed(42)
    x = torch.randn(3, PHASES, RAW_POINTS)
    model = FivePhasePlainCNN1D()
    logits, features = model(x, return_features=True)
    assert logits.shape == (3, CLASSES) and features.shape == (3, 128)
    logits.square().mean().backward()
    assert model.stem[0].weight.grad is not None
    assert not any("shortcut" in key for key in model.state_dict())
    print("Plain CNN1D matched-depth smoke PASS",sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    smoke()
