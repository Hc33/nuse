# encoding: UTF-8

import nuse.nn.mobilenetv2
import nuse.nn.fcn
import torch.nn


class UpBlock(torch.nn.Module):
    def __init__(self, bottom_in_channels, bypass_in_channels, out_channels):
        super().__init__()
        self.upsample = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nuse.nn.mobilenetv2.InvertedResidual(bottom_in_channels, bypass_in_channels, 1, expand_ratio=1),
            nuse.nn.mobilenetv2.InvertedResidual(bypass_in_channels, bypass_in_channels, 1, expand_ratio=6))
        self.merge = torch.nn.Sequential(
            nuse.nn.mobilenetv2.InvertedResidual(2 * bypass_in_channels, bypass_in_channels, 1, expand_ratio=1),
            nuse.nn.mobilenetv2.InvertedResidual(bypass_in_channels, out_channels, 1, expand_ratio=6))

    def forward(self, bottom, bypass):
        print(bottom.size(), bypass.size())
        bottom = self.upsample(bottom)
        return self.merge(torch.cat((bottom, bypass), dim=1))


class MobileUNet(torch.nn.Module):
    def __init__(self, pretrained: str = None):
        super().__init__()
        m2 = nuse.nn.mobilenetv2.MobileNetV2()
        m2.load_state_dict(torch.load(pretrained, 'cpu'))
        features = m2.features
        self.down1 = features[0:2]
        self.down2 = features[2:4]
        self.down3 = features[4:7]
        self.down4 = features[7:14]
        self.down5 = features[14:]
        self.up4 = UpBlock(1280, 96, 96)
        self.up3 = UpBlock(96, 32, 32)
        self.up2 = UpBlock(32, 24, 24)
        self.up1 = UpBlock(24, 16, 16)
        self.predict = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)

        self.factor = 2

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        up4 = self.up4(down5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)
        return self.predict(up1)
