# encoding: UTF-8

import torch.nn
import torchvision
from nuse.nn.init import recursively_initialize


class CSE(torch.nn.Sequential):
    def __init__(self, num_channels):
        super().__init__(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Conv2d(num_channels, num_channels // 2, 1),
            torch.nn.BatchNorm2d(num_channels // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_channels // 2, num_channels, 1),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.Sigmoid())


class SSE(torch.nn.Sequential):
    def __init__(self, num_channels):
        super().__init__(
            torch.nn.Conv2d(num_channels, 1, 1),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid())


class Attention(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.sse = SSE(num_channels)
        self.cse = CSE(num_channels)

    def forward(self, x):
        return self.sse(x) * x + self.cse(x)* x


class ConvBNReLU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True))

        recursively_initialize(self, torch.nn.Conv2d, torch.nn.init.kaiming_normal_)


class UpsamleBlock(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__(
            torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            ConvBNReLU(in_channels, out_channels, 3, 1, 1))


class UpBlock(torch.nn.Module):
    def __init__(self, down_channels, bypass_channels, out_channels, use_attention=False):
        super().__init__()
        self.upsample = UpsamleBlock(down_channels, bypass_channels)
        self.fuse = ConvBNReLU(bypass_channels * 2, out_channels, 3, 1, 1)

        if use_attention:
            self.attention = Attention(out_channels)
        else:
            self.attention = torch.nn.Identity()

        self.factor = 1
        self.bottom_scale = 16

    def forward(self, down, bypass):
        up = self.upsample(down)
        x = torch.cat((bypass, up), dim=1)
        x = self.fuse(x)
        return self.attention(x)


class ResUNet(torch.nn.Module):
    def __init__(self, use_attention):
        super().__init__()
        r = torchvision.models.resnet50(pretrained=True)
        r.conv1.stride = 1, 1
        r.conv1.padding = 3, 3
        self.conv = torch.nn.Sequential(r.conv1, r.bn1, torch.nn.ReLU(inplace=True))
        self.down1 = torch.nn.Sequential(r.maxpool, r.layer1)
        self.down2 = r.layer2
        self.down3 = r.layer3
        self.down4 = r.layer4
        self.up4 = UpBlock(2048, 1024, 64, use_attention=use_attention)
        self.up3 = UpBlock(64, 512, 64, use_attention=use_attention)
        self.up2 = UpBlock(64, 256, 64, use_attention=use_attention)
        self.x8 = UpsamleBlock(64, 64, scale_factor=8)
        self.x4 = UpsamleBlock(64, 64, scale_factor=4)
        self.x2 = UpsamleBlock(64, 64, scale_factor=2)
        self.predict = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, 3, 1, 1),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x):
        first = self.conv(x)
        down1 = self.down1(first)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        feature = torch.cat((
            self.x8(up4),
            self.x4(up3),
            self.x2(up2),
            first), dim=1)
        return self.predict(feature)


if __name__ == '__main__':
    def __debug_nuse_nn_resunet__():
        import torchstat
        x = torch.zeros(1, 3, 64, 64)
        m = ResUNet(use_attention=False)
        torchstat.stat(m, (3, 64, 64))
        print(m(x).size())

    __debug_nuse_nn_resunet__()
