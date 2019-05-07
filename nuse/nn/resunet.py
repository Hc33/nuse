# encoding: UTF-8

import torch.nn
import torchvision
from nuse.nn.common import UpBlock, hyper_column, ConvBNReLU


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
        self.up1 = UpBlock(64, 64, 64, use_attention=use_attention)
        self.predict = torch.nn.Sequential(
            ConvBNReLU(256, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 3, 1))

        self.factor = 1
        self.bottom_scale = 16

    def forward(self, x):
        first = self.conv(x)
        down1 = self.down1(first)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, first)
        return self.predict(hyper_column(up4, up3, up2, up1))


if __name__ == '__main__':
    def __debug_nuse_nn_resunet__():
        import torchstat
        x = torch.zeros(1, 3, 64, 64)
        m = ResUNet(use_attention=False)
        torchstat.stat(m, (3, 64, 64))
        print(m(x).size())

    __debug_nuse_nn_resunet__()
