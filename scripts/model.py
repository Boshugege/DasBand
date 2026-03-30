# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - import guard
    torch = None

    class _NNStub:
        Module = object

    nn = _NNStub()


class MixedConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        branch = max(8, out_ch // 3)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, branch, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, branch, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, branch, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch * 3, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.fuse(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = MixedConvBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        dt = skip.shape[-2] - x.shape[-2]
        dc = skip.shape[-1] - x.shape[-1]
        if dt != 0 or dc != 0:
            x = nn.functional.pad(x, [0, max(0, dc), 0, max(0, dt)])
            x = x[:, :, : skip.shape[-2], : skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DASBandUNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        self.enc1 = MixedConvBlock(in_ch, base_ch, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = MixedConvBlock(base_ch, base_ch * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = MixedConvBlock(base_ch * 2, base_ch * 4, dropout=dropout)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, dropout=dropout)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, dropout=dropout)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        xu = self.up2(xb, x2)
        xu = self.up1(xu, x1)
        return self.head(xu)
