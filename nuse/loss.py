# encoding: UTF-8

import torch
from nuse.trainer import MultiLoss
from nuse.nn.functional.lovasz import lovasz_hinge


def meta_criterion(fn):
    def __criterion__(hypot, label, *args, **kwargs):
        h_outside, h_boundary, h_inside = map(lambda t: t.unsqueeze(1), torch.split(hypot, 1, dim=1))
        y_outside, y_boundary, y_inside = map(lambda t: t.unsqueeze(1), torch.split(label.float(), 1, dim=1))
        loss_outside = fn(h_outside, y_outside, *args, **kwargs)
        loss_boundary = fn(h_boundary, y_boundary, *args, **kwargs)
        loss_inside = fn(h_inside, y_inside, *args, **kwargs)
        loss = loss_outside + loss_boundary + loss_inside
        return MultiLoss(outside=loss_outside, boundary=loss_boundary, inside=loss_inside, overall=loss)

    return __criterion__


@meta_criterion
def bce_criterion(h, y, k=None):
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


def dice_criterion(h, y):
    batch_size, eps = h.size(0), 1e-5
    intersection = (h * y).view(batch_size, -1).sum(1)
    union = h.view(batch_size, -1).sum(1) + y.view(batch_size, -1).sum(1)
    loss = 1 - 2 * (intersection + eps) / (union + eps)
    return loss.mean()


def lovasz_criterion(h, y):
    h, y = h.squeeze(1), y.squeeze(1)
    return lovasz_hinge(h, y)


def symmetric_lovasz_criterion(h, y):
    h, y = h.squeeze(1), y.squeeze(1)
    return (lovasz_hinge(h, y) + lovasz_hinge(-h, 1 - y)) / 2


def get_criterion(name):
    name = name.lower()
    if name == 'bce':
        return torch.sigmoid, torch.sigmoid, bce_criterion
    elif name == 'dice':
        return torch.sigmoid, torch.sigmoid, dice_criterion
    elif name == 'lovasz':
        return None, torch.sigmoid, lovasz_criterion
    elif name == 'symmetric_lovasz':
        return None, torch.sigmoid, symmetric_lovasz_criterion
    else:
        raise NameError('Unknown criterion {!r}. Choose from {{bce, dice, lovasz, symmetric_lovasz}}'.format(name))
