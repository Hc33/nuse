# encoding: UTF-8

import torch


def dice(h, y, eps=1e-5):
    intersection = (h * y).sum()
    union = h.sum() + y.sum()
    return 1 - 2 * intersection / union


def bce_loss(h, y, k=None):
    from torch.nn.functional import binary_cross_entropy_with_logits
    if k is None:
        return binary_cross_entropy_with_logits(h, y)
    batch_size = h.size(0)
    bce = binary_cross_entropy_with_logits(h, y, reduction='none').view(batch_size, -1)
    bce = torch.sort(bce, dim=1, descending=True)[0]
    worst = bce[:, :k].mean()
    with torch.no_grad():
        mask = torch.zeros_like(bce)
        mask[:, k:].fill_(k / (bce.size(1) - k))
        mask = torch.bernoulli(mask).byte()
    random = torch.masked_select(bce, mask).mean()
    return (worst + random) / 2


