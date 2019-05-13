# encoding: UTF-8

"""
MoNuSeg packager
"""

import multiprocessing.pool
import os
import shutil
import sys
import zipfile

import PIL.Image
import cv2
import numpy as np
import staintools
import torch

from MoNuSeg.traits import *
from MoNuSeg.parser import Annotation


def calculate_mean_std():
    rs, gs, bs = [], [], []
    for tissue in get_training_tissues():
        print('[Mean/Std]', tissue.patient_id)
        im = cv2.imread(f'./Tissue/norm_{tissue.patient_id}.png').astype(np.float32) / 255
        b, g, r = im[:, :, 0].reshape(-1), im[:, :, 1].reshape(-1), im[:, :, 2].reshape(-1)
        rs.append(r)
        gs.append(g)
        bs.append(b)
    r = np.concatenate(rs)
    g = np.concatenate(gs)
    b = np.concatenate(bs)
    mean = [r.mean(), g.mean(), b.mean()]
    std = [r.std(), g.std(), b.std()]
    print(f'mean={mean}')
    print(f'std={std}')
    return mean, std


def build_normalizer():
    standard = get_stain_normalization_target()  # type: Tissue
    target = staintools.read_image(f'./Tissue/{standard.patient_id}.tif')
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    return normalizer


def _normalize_fn(tissue: Tissue):
    print('[StainNormalize]', tissue.patient_id)
    normalizer = build_normalizer()
    image = staintools.read_image(f'./Tissue/{tissue.patient_id}.tif')
    image = staintools.LuminosityStandardizer.standardize(image)
    image = normalizer.transform(image)
    cv2.imwrite(f'./Tissue/norm_{tissue.patient_id}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def normalize():
    pool = multiprocessing.pool.Pool()
    pool.map(_normalize_fn, get_all_tissues())


def on_region(region):
    box = region[:, 0].min(), region[:, 1].min(), region[:, 0].max(), region[:, 1].max()
    return box, region


def render(height: int, width: int, anno: Annotation):
    boundary = np.zeros((height, width), dtype=np.uint8)
    inside = np.zeros((height, width), dtype=np.uint8)
    canvas = np.zeros((height, width), dtype=np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for region in anno.regions:
        if region.area_px < 32:
            continue
        canvas[...] = 0
        cv2.fillPoly(canvas, [region.contour.round().astype(np.int32)], 1)
        dilated = cv2.dilate(canvas, kernel)
        eroded = cv2.erode(canvas, kernel)
        mask = dilated > 0
        boundary[mask] = 0
        boundary += dilated - eroded
        inside[mask] = 0
        inside += eroded
    outside = ((inside + boundary) == 0).astype(np.uint8)
    return np.stack((outside, boundary, inside), axis=0)  # type: np.ndarray


def unpack_zip(filename):
    monuseg = zipfile.ZipFile(filename)  # type: zipfile.ZipFile
    for tissue in get_all_tissues():
        print(f'[Unzip] {tissue.patient_id}')
        tif_name = f'MoNuSeg Training Data/Tissue images/{tissue.patient_id}.tif'
        xml_name = f'MoNuSeg Training Data/Annotations/{tissue.patient_id}.xml'
        with open(f'Tissue/{tissue.patient_id}.tif', 'wb') as fd:
            fd.write(monuseg.read(tif_name))
        with open(f'Annotations/{tissue.patient_id}.xml', 'wb') as fd:
            fd.write(monuseg.read(xml_name))
    monuseg.close()


def create_tmp_dir():
    if os.path.isdir('Tissue') or os.path.isdir('Annotations'):
        raise FileExistsError('./Tissue/ or ./Annotations/ already exists. Remove them and try again.')
    os.makedirs('Tissue', exist_ok=False)
    os.makedirs('Annotations', exist_ok=False)


def remove_tmp_dir():
    shutil.rmtree('Tissue')
    shutil.rmtree('Annotations')


def pack(pth_filename):
    mean, std = calculate_mean_std()
    images = {}
    annotations = {}
    labels = {}

    for tissue in get_all_tissues():
        print('[Pack]', tissue.patient_id)

        image = PIL.Image.open(f'./Tissue/norm_{tissue.patient_id}.png')
        annotation = Annotation.from_xml_file(f'./Annotations/{tissue.patient_id}.xml')
        label = render(image.height, image.width, annotation)

        images[tissue.patient_id] = image
        annotations[tissue.patient_id] = annotation
        labels[tissue.patient_id] = label

    torch.save(dict(images=images, labels=labels, annotations=annotations, mean=mean, std=std), pth_filename)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python', sys.argv[0], 'ZIP_FILENAME', 'PTH_FILNAME')
        exit(-1)
    _, monuseg_zip, output_pth = sys.argv
    create_tmp_dir()
    unpack_zip(monuseg_zip)
    normalize()
    pack(output_pth)
    remove_tmp_dir()
