# encoding: UTF-8


import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from nuse.utils.cnn3_decoder import decode_batch


class DepthConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=bias))


class UNetDownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2, depthwise=True):
        Conv2d = DepthConv2d if depthwise else nn.Conv2d
        super().__init__(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels))


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise=True):
        super().__init__()
        Conv2d = DepthConv2d if depthwise else nn.Conv2d
        self.up = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, bias=False, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=in_channels))
        self.merge = nn.Sequential(
            Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels))

    def forward(self, bypass, bottom):
        bottom = interpolate(bottom, scale_factor=2, mode='bilinear', align_corners=True)
        top = self.up(bottom)
        return self.merge(torch.cat((bypass, top), dim=1))


class FCN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = UNetDownBlock(3, 32, stride=1, depthwise=False)
        self.down1 = UNetDownBlock(32, 64)
        self.down2 = UNetDownBlock(64, 128)
        self.down3 = UNetDownBlock(128, 256)
        self.down4 = UNetDownBlock(256, 512)
        self.down5 = UNetDownBlock(512, 512)
        self.up4 = UNetUpBlock(512, 256)
        self.up3 = UNetUpBlock(256, 128)
        self.up2 = UNetUpBlock(128, 64)
        self.up1 = UNetUpBlock(64, 32)
        self.out = UNetUpBlock(32, 32)
        self.predict = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

        self.factor = 1
        self.bottom_scale = 32

        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, image):
        conv = self.conv(image)
        down1 = self.down1(conv)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        up4 = self.up4(down4, down5)
        up3 = self.up3(down3, up4)
        up2 = self.up2(down2, up3)
        up1 = self.up1(down1, up2)
        out = self.out(conv, up1)
        return self.predict(out)

    @staticmethod
    def output_transform(x, y, y_pred, regions):
        return decode_batch(y_pred[:, 1], y_pred[:, 2])


if __name__ == '__main__':
    def __debug_nuse_nn_fcn__():
        import torchstat
        m = FCN()
        torchstat.stat(m, (3, 64, 64))
        x = torch.zeros(1, 3, 64, 64)
        assert x.size(2) == m(x).size(2)
        assert x.size(3) == m(x).size(3)

    __debug_nuse_nn_fcn__()
