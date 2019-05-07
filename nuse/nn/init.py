# encoding: UTF-8


import torch.nn


def recursively_initialize(parent: torch.nn.Module, module_type, weighe_init_fn, bias_init_fn=None):
    for module in parent.modules():
        if isinstance(module, module_type):
            if module.weight is not None:
                weighe_init_fn(module.weight)
            if module.bias is not None and bias_init_fn is not None:
                bias_init_fn(module.bias)
