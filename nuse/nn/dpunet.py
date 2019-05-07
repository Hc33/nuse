# encoding: UTF-8


import torch
import torch.nn
import torch.nn.init

import pretrainedmodels.models.dpn as dpn
from nuse.nn.common import UpBlock, ConvBNReLU, hyper_column
from nuse.nn.init import recursively_initialize
from torch.utils.checkpoint import checkpoint
import functools


@functools.wraps(checkpoint)
def checkpoint_with_hyper_column(fn):
    def __fn_with_hyper_column__(*args):
        return fn(hyper_column(*args))

    def __optimizied__(*args):
        return checkpoint(__fn_with_hyper_column__, *args)

    return __optimizied__


class DPDecoder(UpBlock):
    def forward(self, down, bypass):
        if torch.is_tensor(bypass):
            return super().forward(down, bypass)
        else:
            return checkpoint(self.forward_efficient, down, *bypass)

    def forward_efficient(self, down, *bypass):
        bypass = torch.cat(bypass, dim=1)
        return super().forward(down, bypass)


class DPUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = dpn.dpn92().features
        block = m.conv1_1  # type: dpn.InputBlock
        block.conv.stride = 1, 1
        self.encoder1 = torch.nn.Sequential(block.conv, block.bn, block.act)
        self.encoder2 = torch.nn.Sequential(block.pool, m.conv2_1, m.conv2_2, m.conv2_3)
        self.encoder3 = torch.nn.Sequential(m.conv3_1, m.conv3_2, m.conv3_3, m.conv3_4)
        self.encoder4 = torch.nn.Sequential(*[getattr(m, f'conv4_{i + 1}') for i in range(20)])
        self.encoder5 = torch.nn.Sequential(m.conv5_1, m.conv5_2, m.conv5_3, m.conv5_bn_ac)
        self.decoder4 = DPDecoder(2688, 1552, 64)
        self.decoder3 = DPDecoder(64, 704, 64)
        self.decoder2 = DPDecoder(64, 336, 64)
        self.decoder1 = DPDecoder(64, 64, 64)

        self.predict = torch.nn.Sequential(
            ConvBNReLU(256, 64, 3, 1, 1),
            torch.nn.Conv2d(64, 3, 1))

        self.predict_efficient = checkpoint_with_hyper_column(self.predict.forward)
        recursively_initialize(self.predict, torch.nn.Conv2d, torch.nn.init.kaiming_normal_)

    def forward(self, x):
        e1 = self.encoder1(x)   # /1    64
        e2 = self.encoder2(e1)  # /2   256,  80 ->  336
        e3 = self.encoder3(e2)  # /4   512, 192 ->  704
        e4 = self.encoder4(e3)  # /8  1024, 528 -> 1552
        e5 = self.encoder5(e4)  # /16 2048, 640 -> 2688
        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return self.predict_efficient(d4, d3, d2, d1)


if __name__ == '__main__':
    def __debug_nuse_nn_dpunet__():
        import torchstat
        x = torch.zeros(1, 3, 64, 64)
        m = DPUNet()
        torchstat.stat(m, (3, 64, 64))
        print(m(x).size())

    __debug_nuse_nn_dpunet__()
