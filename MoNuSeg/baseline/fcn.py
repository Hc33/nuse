# encoding: UTF-8


import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class UNetDownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2),
            nn.ReLU(inplace=True))
        self.merge = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

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
        self.down2 = UNetDownBlock(32, 64)
        self.down3 = UNetDownBlock(64, 128)
        self.down4 = UNetDownBlock(128, 128)
        self.bottom = UNetDownBlock(128, 128)
        self.up4 = UNetUpBlock(128, 128)
        self.up3 = UNetUpBlock(128, 64)
        self.up2 = UNetUpBlock(64, 32)
        self.up1 = UNetUpBlock(32, 32)
        self.predict = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

    def forward(self, image):
        down1 = self.down1(image)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        bottom = self.bottom(down4)
        up4 = self.up4(down4, bottom)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        feature = self.up1(down1, up2)
        return self.predict(feature)


fcn = FCN()
print(fcn(torch.zeros(1, 3, 256, 256)).size())
