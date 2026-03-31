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
            nn.ReLU(inplace=True)
        )
        
        # 添加通道注意力机制 (Squeeze-and-Excitation, SE Block)
        # 允许网络在全局范围内"重新加权"不同通道的特征，对抗强烈的背景噪声
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(1, out_ch // 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_ch // 4), out_ch, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x = self.fuse(x)
        
        # 施加注意力权重
        se_weight = self.se(x)
        x = x * se_weight
        
        return self.dropout(x)


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
    def __init__(self, in_ch: int, base_ch: int = 64, dropout: float = 0.15):
        super().__init__()
        # 加深了一层网络，扩大感受野，消耗更多算力但能捕捉更长的时间依赖
        self.enc1 = MixedConvBlock(in_ch, base_ch, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = MixedConvBlock(base_ch, base_ch * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = MixedConvBlock(base_ch * 2, base_ch * 4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = MixedConvBlock(base_ch * 4, base_ch * 8, dropout=dropout)
        
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, dropout=dropout)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, dropout=dropout)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, dropout=dropout)
        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        
        xb = self.bottleneck(self.pool3(x3))
        
        xu = self.up3(xb, x3)
        xu = self.up2(xu, x2)
        xu = self.up1(xu, x1)
        return self.head(xu)
