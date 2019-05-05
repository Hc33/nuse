# encoding: UTF-8

import cv2
import torch
import numpy as np


def decode(boundary, nucleus):
    nucleus[boundary > 0.25] = 0
    total_nucleus, nucleus = cv2.connectedComponents(nucleus)
    result = np.zeros(boundary.shape, np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    boundary_mean = boundary[boundary >= 0.25].mean()
    current = np.zeros((1 + total_nucleus,))
    canvas = np.zeros(boundary.shape)
    canvas_rgb = np.zeros(boundary.shape)
    for i in range(1, total_nucleus + 1):
        nuclei = (nucleus == i).astype(np.uint8)
        count = 0
        if not nuclei.any():
            continue
        while current[i] < boundary_mean and count < 2:
            dilated = cv2.dilate(nuclei, kernel)
            count += 1
            diff = (dilated - nuclei) * boundary
            current[i] = diff[diff > 0].mean()
            nuclei = dilated
        canvas += nuclei
    return canvas


def decode_image(boundary_filename: str, nucleus_filename: str, result_filename: str):
    boundary = cv2.imread(boundary_filename)[:, :, 0].astype(np.float32) / 255
    nucleus  = ((cv2.imread(nucleus_filename)[:, :, 0].astype(np.float32) / 255) >= 0.5).astype(np.uint8)
    nucleus  = decode(boundary, nucleus) * 255
    cv2.imwrite(result_filename, nucleus)


def decode_tensor(h_boundary, h_inside):
    boundary = h_boundary[0].cpu().numpy()
    nucleus = (h_inside[0] >= 0.5).cpu().byte().numpy()
    return (torch.from_numpy(decode(boundary, nucleus)) >= 0.5).byte()


def decode_batch(h_boundary, h_inside):
    batch = []
    for b, n in zip(h_boundary, h_inside):
        batch.append(decode_tensor(b, n))
    return torch.stack(batch, dim=0)
