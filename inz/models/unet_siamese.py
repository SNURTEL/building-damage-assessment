import torch
import torch.nn as nn
from torch import Tensor


class DoubleConvS(nn.Module):
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


class DownS(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConvS(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)  # type: ignore[no-any-return]


class UpS(nn.Module):
    def __init__(self, in_channels: int, in_channels_res: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvS(in_channels_res + out_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # type: ignore[no-any-return]


class OutConvS(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConvS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)  # type: ignore[no-any-return]


class UNetSiamese(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super(UNetSiamese, self).__init__()
        self.n_channels = in_channels

        self.inc = DoubleConvS(in_channels, 128)
        self.down1 = DownS(128, 256)
        self.down2 = DownS(256, 512)
        self.down3 = DownS(512, 1024)
        self.down4 = DownS(1024, 2048)
        self.up1 = UpS(4096, 2048, 1024)
        self.up2 = UpS(1024, 1024, 512)
        self.up3 = UpS(512, 512, 256)
        self.up4 = UpS(256, 256, 128)
        self.outc = OutConvS(128, out_channels)

    def forward(self, xa: Tensor, xb: Tensor) -> Tensor:
        x1a = self.inc(xa)
        x2a = self.down1(x1a)
        x3a = self.down2(x2a)
        x4a = self.down3(x3a)
        x5a = self.down4(x4a)
        x1b = self.inc(xb)
        x2b = self.down1(x1b)
        x3b = self.down2(x2b)
        x4b = self.down3(x3b)
        x5b = self.down4(x4b)
        x = self.up1(torch.cat((x5a, x5b), dim=1), torch.cat((x4a, x4b), dim=1))
        x = self.up2(x, torch.cat((x3a, x3b), dim=1))
        x = self.up3(x, torch.cat((x2a, x2b), dim=1))
        x = self.up4(x, torch.cat((x1a, x1b), dim=1))
        x = self.outc(x)
        return x  # type: ignore[no-any-return]
