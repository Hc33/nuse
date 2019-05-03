# encoding: UTF-8

import torch


def iou(x, y):
    i = (x * y).sum().item()
    u = ((x + y) > 0).sum().item()
    if u == 0:
        return 0
    return i / u


def accuracy(label, h_outside, h_boundary, h_inside):
    with torch.no_grad():
        h_outside  = (h_outside  >= 0.5).long()
        h_boundary = (h_boundary >= 0.5).long()
        h_inside   = (h_inside   >= 0.5).long()
        ao = iou(h_outside,  label[:, 0:1]) # (h_outside  == label[:, 0:1]).sum().item() / h_outside.numel()
        ab = iou(h_boundary, label[:, 1:2]) # (h_boundary == label[:, 1:2]).sum().item() / h_boundary.numel()
        ai = iou(h_inside,   label[:, 2:3]) # (h_inside   == label[:, 2:3]).sum().item() / h_inside.numel()
        return (ao + ab + ai) / 3, ao, ab, ai
