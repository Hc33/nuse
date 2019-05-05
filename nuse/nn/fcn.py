# encoding: UTF-8


import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from nuse.utils.cnn3_decoder import decode_batch
import nuse.nn.functional.helpers as helpers


class DepthConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels))


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, depthwise=True):
        Conv2d = DepthConv2d if depthwise else nn.Conv2d
        super().__init__()
        self.left = Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=stride)
        self.right = Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.left(x)
        return x + self.right(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise=True):
        super().__init__()
        Conv2d = DepthConv2d if depthwise else nn.Conv2d
        self.up = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, bias=False, padding=1)
        self.c1 = Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.c2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, bypass, bottom):
        bottom = interpolate(bottom, scale_factor=2, mode='bilinear', align_corners=True)
        top = self.up(bottom)
        x = torch.cat((bypass, top), dim=1)
        x = self.c1(x)
        return x + self.c2(x)


class FCN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.down = nn.ModuleList([
            UNetDownBlock(3, 32, stride=1, depthwise=False),
            UNetDownBlock(32, 64),
            UNetDownBlock(64, 128),
            UNetDownBlock(128, 256),
            UNetDownBlock(256, 512),
            UNetDownBlock(512, 512),
            UNetDownBlock(512, 512)
        ])
        self.up = nn.ModuleList([
            UNetUpBlock(512, 512),
            UNetUpBlock(512, 256),
            UNetUpBlock(256, 128),
            UNetUpBlock(128, 64),
            UNetUpBlock(64, 32),
            UNetUpBlock(32, 32),
        ])
        self.predict = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)

        self.factor = 1

        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, image):
        features = helpers.unet_scan(image, self.down)
        feature = helpers.unet_fold(features, self.up)
        return self.predict(feature)

    @staticmethod
    def output_transform(x, y, y_pred, regions):
        return decode_batch(y_pred[:, 1], y_pred[:, 2])


if __name__ == '__main__':
    def __debug_fcn__():
        import torchstat
        m = FCN()
        torchstat.stat(m, (3, 64, 64))
        x = torch.zeros(1, 3, 64, 64)
        assert x.size(2) == m(x).size(2)
        assert x.size(3) == m(x).size(3)


    __debug_fcn__()
