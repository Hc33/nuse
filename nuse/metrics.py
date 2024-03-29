# encoding: UTF-8

import statistics

import cv2
import numpy as np
import torch
from ignite.metrics import Metric

from nuse.trainer import Prediction
from nuse.utils.cnn3_decoder import decode


def iou(x, y):
    i = (x * y).sum().item()
    u = ((x + y) > 0).sum().item()
    if u == 0:
        return 0
    return i / u


def accuracy(label, h_outside, h_boundary, h_inside):
    with torch.no_grad():
        h_outside = (h_outside >= 0.5).long()
        h_boundary = (h_boundary >= 0.5).long()
        h_inside = (h_inside >= 0.5).long()
        ao = iou(h_outside, label[:, 0:1])  # (h_outside  == label[:, 0:1]).sum().item() / h_outside.numel()
        ab = iou(h_boundary, label[:, 1:2])  # (h_boundary == label[:, 1:2]).sum().item() / h_boundary.numel()
        ai = iou(h_inside, label[:, 2:3])  # (h_inside   == label[:, 2:3]).sum().item() / h_inside.numel()
        return (ao + ab + ai) / 3, ao, ab, ai


def area_filter(binary, threshold=32):
    num, comp_map = cv2.connectedComponents(binary)
    for comp_value in range(1, num):
        mask = comp_map == comp_value
        if mask.sum() < threshold:
            comp_map[mask] = 0
    return np.clip(comp_map, a_min=0, a_max=1).astype(np.uint8)


def union_box(lhs, rhs):
    return min(lhs[0], rhs[0]), min(lhs[1], rhs[1]), max(lhs[2], rhs[2]), max(lhs[3], rhs[3])


def collision_test(lhs, rhs):
    cx = lhs[2] >= rhs[0] and rhs[2] >= lhs[0]
    cy = lhs[3] >= rhs[1] and rhs[3] >= lhs[1]
    if cx and cy:
        return union_box(lhs, rhs)
    return None


def masked_iou(lhs, rhs):
    i = (lhs & rhs).sum()
    u = (lhs | rhs).sum()
    if u == 0:
        return 0, i, u
    return i / u, i, u


def aji_metric(binary: np.ndarray, regions: [((int, int, int, int), np.ndarray)]):
    # Initialize overall correct and union pixel counts
    correct_pixels, union_pixels = 0, 0

    # Render ground truth
    truth = np.zeros_like(binary, dtype=np.int32)  # type: np.ndarray
    for value, (_, region) in enumerate(regions, start=1):
        cv2.drawContours(truth, [region], contourIdx=-1, color=value, thickness=cv2.FILLED)

    # solve CCL
    num_components, component_map = cv2.connectedComponents(binary)  # type: int, np.ndarray
    component_boxes = []
    for i in range(1, num_components):
        component = component_map == i
        x, y, w, h = cv2.boundingRect(component.astype(np.uint8))
        box = x, y, x + w, y + h
        component_boxes.append((i, box))

    # for each ground truth nuclei `region` do
    for region_value, (region_box, _) in enumerate(regions, start=1):
        area_cache = [(None, None)] * len(component_boxes)
        iou_record = np.zeros((len(component_boxes),), dtype=np.float)

        # accelerate IoU via collision detect
        for idx in range(len(component_boxes)):
            value, detected_box = component_boxes[idx]
            if value is None:
                continue
            union = collision_test(region_box, detected_box)
            if union is not None:
                x1, y1, x2, y2 = union
                detected = component_map[y1:y2 + 1, x1:x2 + 1] == value
                selected = truth[y1:y2 + 1, x1:x2 + 1] == region_value
                iou_record[idx], intersect_area, union_area = masked_iou(detected, selected)
                area_cache[idx] = intersect_area, union_area

        # best_match_idx <- argmax(k, |Gi/\Sk|/|Gi\/Sk|)
        best_match_idx = np.argmax(iou_record).item()

        # update counters
        if iou_record[best_match_idx] > 0:
            component_boxes[best_match_idx] = None, None
            c, u = area_cache[best_match_idx]
            correct_pixels += c
            union_pixels += u

    # for Each segmented nucleus do
    for value, _ in component_boxes:
        if value is not None:
            union_pixels += (component_map == value).sum()

    # A <- C / U
    return correct_pixels / union_pixels, correct_pixels, union_pixels


class AJIMetric(Metric):
    def __init__(self, threshold=0.5, padding=12, target_size=1000):
        self.aji_for_all = []
        self.threshold = threshold
        self.padding = padding
        self.target_size = target_size
        self.crop = slice(self.padding, self.padding + self.target_size)
        super().__init__()

    def reset(self):
        self.aji_for_all.clear()

    def update(self, predictions: Prediction):
        batch_size = predictions.prediction.size(0)
        for i in range(batch_size):
            boundary, inside = predictions.prediction[i, 1:, self.crop, self.crop]
            regions = predictions.regions[i]
            boundary = (boundary > self.threshold).int()
            inside = (((inside > self.threshold).int() - boundary) > self.threshold).byte().cpu().numpy()
            boundary = boundary.byte().cpu().numpy()
            inside = area_filter(inside, threshold=32)
            inside = cv2.erode(inside, np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=np.uint8))
            decoded = decode(boundary, inside).astype(np.uint8)
            result = area_filter(decoded, threshold=32)
            aji, _, _ = aji_metric(result, regions)
            self.aji_for_all.append(aji)

    def compute(self):
        return statistics.mean(self.aji_for_all)
