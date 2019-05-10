# encoding: UTF-8

from lxml import etree
from zipfile import ZipFile
from MoNuSeg.traits import Tissue
from dataclasses import dataclass
import numpy as np


@dataclass
class BoundingBox:
    x: float
    y: float
    h: float
    w: float

    @property
    def area(self) -> float:
        return self.h * self.w

    def to_cv2(self) -> (float, float, float, float):
        return self.x, self.y, self.w, self.h

    @staticmethod
    def from_cv2(bbox) -> 'BoundingBox':
        return BoundingBox(**bbox)


@dataclass
class Region:
    box: BoundingBox
    area_px: float
    area_um: float
    length_px: float
    length_um: float
    contour: np.ndarray


@dataclass
class Annotation:
    regions: [Region]

    @staticmethod
    def from_zip(zipfile: ZipFile, tissue: Tissue) -> 'Annotation':
        xml_string = zipfile.read(f'Annontations/{tissue.patient_id}.xml').decode()
        return Annotation.from_xml_string(xml_string)

    @staticmethod
    def from_xml_file(filename: str) -> 'Annotation':
        with open(filename) as fd:
            xml_string = fd.read()
        return Annotation.from_xml_string(xml_string)

    @staticmethod
    def from_xml_string(xml_string: str) -> 'Annotation':
        return Annotation(regions=parse_annotation(xml_string))


def parse_region(region):
    attributes = region.attrib
    vertices = region.find('Vertices')
    contour = np.array([[float(v.attrib['X']), float(v.attrib['Y'])] for v in vertices])
    x, y = contour[:, 0].min().item(), contour[:, 1].min().item()
    w, h = contour[:, 0].max().item(), contour[:, 1].max().item()
    w -= x
    h -= y
    bbox = BoundingBox(x, y, w, h)
    pixel_length = float(attributes['Length'])
    pixel_area = float(attributes['Area'])
    real_length = float(attributes['LengthMicrons'])
    real_area = float(attributes['AreaMicrons'])
    return Region(contour=contour, box=bbox,
                  area_px=pixel_area, area_um=real_area, length_px=pixel_length, length_um=real_length)


def parse_annotation(xml_string: str) -> [Region]:
    root = etree.fromstring(xml_string)
    xml_objects = root.find('Annotation').find('Regions').findall('Region')
    regions = [parse_region(r) for r in xml_objects]  # type: [Region]
    regions.sort(key=lambda r: r.area_px)
    return regions
