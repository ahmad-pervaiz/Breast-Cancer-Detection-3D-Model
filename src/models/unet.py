"""
Configurable U-Net for 2D tumor segmentation.

Plain PyTorch (not a third-party pretrained backbone) so every part is
inspectable/controllable per project requirements. Outputs raw logits -
apply sigmoid only at inference/metric time (see src/training/losses.py and
src/inference/predict.py), matching the BCEWithLogitsLoss convention.
"""
from typing import List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, with optional spatial (channel-wise) dropout
    at the end. dropout_p=0.0 (default) is a no-op - only the bottleneck uses
    this in practice (see UNet.__init__)."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))  # zeroes whole feature channels, not individual pixels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, dropout_p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle any off-by-one size mismatch from odd input dimensions.
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Args:
        in_channels: input image channels (1 for grayscale CT).
        out_channels: output mask channels (1 for binary tumor segmentation).
        base_features: channel count after the first conv block; doubles at
            each downsampling stage (32 -> 64 -> 128 -> 256 -> ... ).
        depth: number of downsampling stages.
        bottleneck_dropout_p: spatial (nn.Dropout2d) dropout applied ONLY at
            the deepest layer (the true bottleneck), not throughout the
            encoder - regularizes the most overfitting-prone layer without
            crippling basic low/mid-level feature extraction. 0.0 = off.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_features: int = 32, depth: int = 4, bottleneck_dropout_p: float = 0.0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        feats: List[int] = [base_features * (2 ** i) for i in range(depth + 1)]

        self.in_conv = DoubleConv(in_channels, feats[0])
        self.downs = nn.ModuleList([
            Down(feats[i], feats[i + 1], dropout_p=bottleneck_dropout_p if i == depth - 1 else 0.0)
            for i in range(depth)
        ])
        self.ups = nn.ModuleList([
            Up(feats[i + 1], feats[i], feats[i]) for i in reversed(range(depth))
        ])
        self.out_conv = nn.Conv2d(feats[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.in_conv(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)

        return self.out_conv(x)  # raw logits, shape (B, out_channels, H, W)


def build_model(cfg: dict) -> UNet:
    return UNet(
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("out_channels", 1),
        base_features=cfg.get("base_features", 32),
        depth=cfg.get("depth", 4),
        bottleneck_dropout_p=cfg.get("bottleneck_dropout_p", 0.0),
    )
