# encoding: UTF-8

"""
MoNuSeg 数据集打包程序
"""

import torch
import cv2
import numpy as np
import PIL.Image
import zipfile
from matplotlib.image import imread
from collections import defaultdict
from lxml import etree
import os
import staintools
import multiprocessing.pool
import shutil
import sys


def calculate_mean_std():
    patient_ids = load_id()
    trainval = patient_ids[0:4] + patient_ids[6:10] + patient_ids[12:16] + patient_ids[18:22]

    rs, gs, bs = [], [], []
    for filename in trainval:
        print('[Mean/Std]', filename)
        im = cv2.imread(f'./Tissue/norm_{filename}.png').astype(np.float32) / 255
        b, g, r = im[:, :, 0].reshape(-1), im[:, :, 1].reshape(-1), im[:, :, 2].reshape(-1)
        rs.append(r)
        gs.append(g)
        bs.append(b)
    r = np.concatenate(rs)
    g = np.concatenate(gs)
    b = np.concatenate(bs)
    mean = [r.mean(), g.mean(), b.mean()] 
    std = [r.std(),  g.std(),  b.std()]
    print(f'mean={mean}')
    print(f'std={std}')
    return mean, std


def build_normalizer():
    target = staintools.read_image('./Tissue/TCGA-G9-6356-01Z-00-DX1.tif')
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)
    return normalizer

def fn(tif):
    print('[StainNormalize]', tif[:-4])
    normalizer = build_normalizer()
    image = staintools.read_image(f'./Tissue/{tif}')
    image = staintools.LuminosityStandardizer.standardize(image)
    image = normalizer.transform(image)
    cv2.imwrite(f'./Tissue/norm_{tif.replace(".tif", ".png")}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def normalize():
    pool = multiprocessing.pool.Pool()

    arglist = []
    for tif in os.listdir('./Tissue'):
        if not tif.endswith('.tif'):
            continue
        arglist.append(tif)

    pool.map(fn, arglist)


def parse_vertex(vertex):
    attr = vertex.attrib
    return float(attr['X']), float(attr['Y'])


def parse_vertics(vertices):
    return np.array([parse_vertex(v) for v in vertices]).round().astype(np.int32)


def parse_region(region):
    return parse_vertics(region.find('Vertices'))


def parse_annotation(filename):
    with open(filename) as fd:
        xml_string = fd.read()
    root = etree.fromstring(xml_string)
    regions = root.find('Annotation').find('Regions')
    return [parse_region(region) for region in regions.findall('Region')]


def on_region(region):
    box = region[:, 0].min(), region[:, 1].min(), region[:, 0].max(), region[:, 1].max()
    return box, region


def render(height, width, xml_filename):
    boundary = np.zeros((height, width), dtype=np.uint8)
    inside = np.zeros((height, width), dtype=np.uint8)
    canvas = np.zeros((height, width), dtype=np.uint8)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    regions = parse_annotation(xml_filename)
    for region in regions:
        canvas[...] = 0
        cv2.fillPoly(canvas, [region], 1)
        dilated = cv2.dilate(canvas, kernel)
        eroded = cv2.erode(canvas, kernel)
        boundary += dilated - eroded
        inside += canvas
    outside = ((inside + boundary) == 0).astype(np.uint8)
    label = np.stack((outside, boundary, inside), axis=0)
    return [on_region(r) for r in regions], label

def load_id():
    xs = []
    with open('details.tsv') as fd:
        for x, *_ in map(str.split, map(str.strip, fd.readlines())):
            xs.append(x)
    return xs

def unpack_zip(filename):
    monuseg = zipfile.ZipFile(filename)  # type: zipfile.ZipFile
    patient_ids = load_id()
    for pid in patient_ids:
        print(f'[Unzip] {pid}')
        tif_name = f'MoNuSeg Training Data/Tissue images/{pid}.tif'
        xml_name = f'MoNuSeg Training Data/Annotations/{pid}.xml'
        with open(f'Tissue/{pid}.tif', 'wb') as fd:
            fd.write(monuseg.read(tif_name))
        with open(f'Annotations/{pid}.xml', 'wb') as fd:
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

    by_patient_id = {}
    by_organ = defaultdict(list)
    by_disase_type = defaultdict(list)


    with open('details.tsv') as fd:
        for idx, line in enumerate(filter(len, map(str.strip, fd.readlines()))):
            print('[Pack]', line)
            patient_id, organ, disease_type = line.split('\t')
            image = PIL.Image.open(f'./Tissue/norm_{patient_id}.png')
            regions, label = render(image.height, image.width, f'./Annotations/{patient_id}.xml')

            by_patient_id[patient_id] = image, label, regions, organ, disease_type
            by_organ[organ].append(patient_id)
            by_disase_type[disease_type].append(patient_id)

    torch.save(dict(by_patient_id=by_patient_id,
                    by_organ=dict(by_organ.items()),
                    by_disase_type=dict(by_disase_type.items()),
                    mean=mean,
                    std=std),
                pth_filename)


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

