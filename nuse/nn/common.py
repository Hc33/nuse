# encoding: UTF-8


import torch
import torch.nn
from nuse.nn.init import recursively_initialize
from torch.nn.functional import interpolate


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


class UpBlock(torch.nn.Module):
    def __init__(self, down_channels, bypass_channels, out_channels, use_attention=False):
        super().__init__()
        self.upsample = UpsamleBlock(down_channels, bypass_channels)
        self.fuse = ConvBNReLU(bypass_channels * 2, out_channels, 3, 1, 1)

        if use_attention:
            self.attention = Attention(out_channels)
        else:
            self.attention = torch.nn.Identity()

    def forward(self, down, bypass):
        up = self.upsample(down)
        x = torch.cat((bypass, up), dim=1)
        x = self.fuse(x)
        return self.attention(x)


def hyper_column(*features, scale_factors=None):
    if scale_factors is None:
        scale_factors = [2 ** i for i in range(len(features) - 1, -1, -1)]
    return torch.cat([interpolate(feature, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                      if scale_factor > 1 else feature
                      for feature, scale_factor in zip(features, scale_factors)], dim=1)

