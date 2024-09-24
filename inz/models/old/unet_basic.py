import torch
import torch.nn as nn
from torch import Tensor


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)  # type: ignore[no-any-return]


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)  # type: ignore[no-any-return]


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # type: ignore[no-any-return]


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)  # type: ignore[no-any-return]


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(UNet, self).__init__()
        self.n_channels = in_channels

        self.inc = DoubleConv(in_channels, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        self.down4 = Down(1024, 2048)
        self.up1 = Up(2048, 1024)
        self.up2 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up4 = Up(256, 128)
        self.outc = OutConv(128, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x  # type: ignore[no-any-return]
