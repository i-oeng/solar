"""Compact U-Net suited to tiled training on a 6 GB GPU."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class FilamentUNet(nn.Module):
    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(2, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bridge = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = UpBlock(base * 16, base * 8, base * 8)
        self.up3 = UpBlock(base * 8, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2, base * 2)
        self.up1 = UpBlock(base * 2, base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        x = self.bridge(self.pool(s4))
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)


class ResNetFilamentUNet(nn.Module):
    """ResNet34 encoder plus a full-resolution intensity/contrast skip path."""

    def __init__(self, encoder_weights: Path | None = None) -> None:
        super().__init__()
        from torchvision.models import resnet34

        self.encoder = resnet34(weights=None)
        if encoder_weights is not None:
            self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu', weights_only=True))
        self.encoder.fc = nn.Identity()
        self.full_resolution = ConvBlock(2, 16)
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.up0 = UpBlock(32, 16, 16)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer('rgb_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full = self.full_resolution(x)
        rgb = (x[:, :1].expand(-1, 3, -1, -1) - self.rgb_mean) / self.rgb_std
        stem = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(rgb)))
        s1 = self.encoder.layer1(self.encoder.maxpool(stem))
        s2 = self.encoder.layer2(s1)
        s3 = self.encoder.layer3(s2)
        s4 = self.encoder.layer4(s3)
        y = self.up4(s4, s3)
        y = self.up3(y, s2)
        y = self.up2(y, s1)
        y = self.up1(y, stem)
        return self.head(self.up0(y, full))


def build_model(architecture: str = 'compact', base: int = 16,
                encoder_weights: Path | None = None) -> nn.Module:
    if architecture == 'compact':
        return FilamentUNet(base=base)
    if architecture == 'resnet34':
        return ResNetFilamentUNet(encoder_weights=encoder_weights)
    raise ValueError(f'Unknown architecture: {architecture}')
