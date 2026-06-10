import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralDifferenceConv2d(nn.Module):
    """Central Difference Convolution usada por CDCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        theta: float = 0.7,
    ):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal
        weight = self.conv.weight
        kernel_diff = weight.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(
            x,
            kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        if out_diff.shape[-2:] != out_normal.shape[-2:]:
            out_diff = F.interpolate(
                out_diff,
                size=out_normal.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return out_normal - self.theta * out_diff


class CDCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, theta: float = 0.7):
        super().__init__()
        self.block = nn.Sequential(
            CentralDifferenceConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CentralDifferenceConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x), inplace=True)


class CDCN(nn.Module):
    """CDCN compacto para inferencia anti-spoofing."""

    def __init__(self, theta: float = 0.7):
        super().__init__()
        self.stem = nn.Sequential(
            CentralDifferenceConv2d(3, 32, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(CDCBlock(32, 64, theta=theta), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(CDCBlock(64, 128, theta=theta), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(CDCBlock(128, 256, theta=theta), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(CDCBlock(256, 384, theta=theta), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x).squeeze(1)
