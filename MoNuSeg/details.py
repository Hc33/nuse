# encoding: UTF-8

import enum
from dataclasses import dataclass

__all__ = ['Organ', 'TumorType', 'Disease', 'Tissue',
           'get_stain_normalization_target', 'get_training_set', 'get_val_set', 'get_all_tissues']


class Organ(enum.Enum):
    BREAST = "Breast"
    KIDNEY = "Kidney"
    LIVER = "Liver"
    PROSTATE = "Prostate"
    BLADDER = "Bladder"
    COLON = "Colon"
    STOMACH = "Stomach"


class TumorType(enum.Enum):
    ADENOMA = 'Adenoma'
    CARCINOMA = 'Carcinoma'
    ADENOCARCINOMA = 'Adenocarcinoma'
    SARCOMA = 'Sarcoma'


@dataclass
class Disease:
    classification: str
    nomenclature: TumorType


@dataclass
class Tissue:
    patient_id: str
    organ: Organ
    disease: Disease

    @property
    def source_site_code(self) -> str:
        return self.patient_id[5:7]


MONUSEG_DETAILS = [
    Tissue("TCGA-A7-A13E-01Z-00-DX1", Organ.BREAST, Disease("Breast Invasive", TumorType.CARCINOMA)),
    Tissue("TCGA-A7-A13F-01Z-00-DX1", Organ.BREAST, Disease("Breast Invasive", TumorType.CARCINOMA)),
    Tissue("TCGA-AR-A1AK-01Z-00-DX1", Organ.BREAST, Disease("Breast Invasive", TumorType.CARCINOMA)),
    Tissue("TCGA-AR-A1AS-01Z-00-DX1", Organ.BREAST, Disease("Breast Invasive", TumorType.CARCINOMA)),
    Tissue("TCGA-E2-A1B5-01Z-00-DX1", Organ.BREAST, Disease("Breast Invasive", TumorType.CARCINOMA)),
    Tissue("TCGA-E2-A14V-01Z-00-DX1", Organ.BREAST, Disease("Breast Invasive", TumorType.CARCINOMA)),
    Tissue("TCGA-B0-5711-01Z-00-DX1", Organ.KIDNEY, Disease("Kidney Renal Clear Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-HE-7128-01Z-00-DX1", Organ.KIDNEY, Disease("Kidney Renal Papillary Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-HE-7129-01Z-00-DX1", Organ.KIDNEY, Disease("Kidney Renal Papillary Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-HE-7130-01Z-00-DX1", Organ.KIDNEY, Disease("Kidney Renal Papillary Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-B0-5710-01Z-00-DX1", Organ.KIDNEY, Disease("Kidney Renal Clear Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-B0-5698-01Z-00-DX1", Organ.KIDNEY, Disease("Kidney Renal Clear Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-18-5592-01Z-00-DX1", Organ.LIVER, Disease("Lung Squamous Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-38-6178-01Z-00-DX1", Organ.LIVER, Disease("Lung", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-49-4488-01Z-00-DX1", Organ.LIVER, Disease("Lung", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-50-5931-01Z-00-DX1", Organ.LIVER, Disease("Lung", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-21-5784-01Z-00-DX1", Organ.LIVER, Disease("Lung Squamous Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-21-5786-01Z-00-DX1", Organ.LIVER, Disease("Lung Squamous Cell", TumorType.CARCINOMA)),
    Tissue("TCGA-G9-6336-01Z-00-DX1", Organ.PROSTATE, Disease("Prostate", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-G9-6348-01Z-00-DX1", Organ.PROSTATE, Disease("Prostate", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-G9-6356-01Z-00-DX1", Organ.PROSTATE, Disease("Prostate", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-G9-6363-01Z-00-DX1", Organ.PROSTATE, Disease("Prostate", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-CH-5767-01Z-00-DX1", Organ.PROSTATE, Disease("Prostate", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-G9-6362-01Z-00-DX1", Organ.PROSTATE, Disease("Prostate", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-DK-A2I6-01A-01-TS1", Organ.BLADDER, Disease("Bladder Urothelial", TumorType.CARCINOMA)),
    Tissue("TCGA-G2-A2EK-01A-02-TSB", Organ.BLADDER, Disease("Bladder Urothelial", TumorType.CARCINOMA)),
    Tissue("TCGA-AY-A8YK-01A-01-TS1", Organ.COLON, Disease("Colon", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-NH-A8F7-01A-01-TS1", Organ.COLON, Disease("Colon", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-KB-A93J-01A-01-TS1", Organ.STOMACH, Disease("Stomach", TumorType.ADENOCARCINOMA)),
    Tissue("TCGA-RD-A8N9-01A-01-TS1", Organ.STOMACH, Disease("Stomach", TumorType.ADENOCARCINOMA)),
]


def get_all_tissues() -> [Tissue]:
    return MONUSEG_DETAILS


def get_stain_normalization_target() -> Tissue:
    return MONUSEG_DETAILS[20]


def get_training_set() -> [Tissue]:
    indies = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21]
    return [MONUSEG_DETAILS[i] for i in indies]


def get_val_set() -> [Tissue]:
    indies = [4, 5, 10, 11, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29]
    return [MONUSEG_DETAILS[i] for i in indies]
