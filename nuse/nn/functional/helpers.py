# encoding: UTF-8


def unet_scan(x, modules):
    xs = []
    for module in modules:
        x = module(x)
        xs.append(x)
    return xs


def unet_fold(features, modules):
    assert len(features) == len(modules) + 1
    x = modules[0](features[-2], features[-1])
    for module, bypass in zip(modules[1:], reversed(features[:-2])):
        x = module(bypass, x)
    return x
