# encoding: UTF-8

import torch.nn
from nuse.nn.fcn import FCN
from nuse.nn.mobile_unet import MobileUNet
from nuse.nn.resunet import ResUNet


def get_model(ns, *args, **kwargs) -> torch.nn.Module:
    name = ns.model  # type: str
    name = name.lower()
    if name == 'fcn':
        return FCN()
    elif name == 'mobileunet':
        return MobileUNet(ns.mobile_v2_pretrained)
    elif name == 'resunet':
        return ResUNet()
    else:
        raise KeyError('Unknown model {!r}. Choose one from {{FCN, MobileUNet}}'.format(ns.model))
