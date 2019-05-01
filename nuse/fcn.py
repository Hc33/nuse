# encoding: UTF-8


import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class UNetDownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2))

        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=in_channels))
        self.merge = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels))

        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, bypass, bottom):
        bottom = interpolate(bottom, scale_factor=2, mode='bilinear', align_corners=True)
        top = self.up(bottom)
        return self.merge(torch.cat((bypass, top), dim=1))


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetDownBlock(3, 32)
        self.down2 = UNetDownBlock(32, 32)
        self.down3 = UNetDownBlock(32, 64)
        self.down4 = UNetDownBlock(64, 64)
        self.down5 = UNetDownBlock(64, 64)
        self.bottom = UNetDownBlock(64, 64)
        self.up5 = UNetUpBlock(64, 64)
        self.up4 = UNetUpBlock(64, 64)
        self.up3 = UNetUpBlock(64, 32)
        self.up2 = UNetUpBlock(32, 32)
        self.up1 = UNetUpBlock(32, 32)
        self.predict = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def forward(self, image):
        down1 = self.down1(image)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        bottom = self.bottom(down5)
        up5 = self.up5(down5, bottom)
        up4 = self.up4(down4, up5)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        feature = self.up1(down1, up2)
        return self.predict(feature)

