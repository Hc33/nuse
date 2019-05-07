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


class UpBlock(torch.nn.Module):
    def __init__(self, down_channels, bypass_channels, out_channels, use_attention=False):
        super().__init__()
        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            torch.nn.Conv2d(down_channels, bypass_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(bypass_channels),
            torch.nn.ReLU(inplace=True))

        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(bypass_channels * 2, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(bypass_channels),
            torch.nn.ReLU(inplace=True))

        if use_attention:
            self.sse = SSE(out_channels)
            self.cse = CSE(out_channels)

        self.use_attention = use_attention
        self.factor = 1
        self.bottom_scale = 16

        recursively_initialize(self, torch.nn.Conv2d, torch.nn.init.kaiming_normal_)

    def forward(self, down, bypass):
        up = self.upsample(down)
        x = torch.cat((bypass, up), dim=1)
        x = self.fuse(x)
        if self.use_attention:
            x = x * self.sse(x) + x * self.cse(x)
        return x


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
        self.up4 = UpBlock(2048, 1024, 1024, use_attention=use_attention)
        self.up3 = UpBlock(1024, 512, 512, use_attention=use_attention)
        self.up2 = UpBlock(512, 256, 256, use_attention=use_attention)
        self.up1 = UpBlock(256, 64, 64, use_attention=use_attention)
        self.predict = torch.nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, x)
        return self.predict(up1)


if __name__ == '__main__':
    def __debug_nuse_nn_resunet__():
        import torchstat
        x = torch.zeros(1, 3, 64, 64)
        m = ResUNet(use_attention=True)
        torchstat.stat(m, (3, 64, 64))
        print(m(x).size())

    __debug_nuse_nn_resunet__()
