# encoding: UTF-8

import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tr
import torchvision.transforms.functional as fn


class ByPass:
    def __call__(self, image):
        return image


class Rotation:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image):
        return fn.rotate(image, self.degree)


class RandomRotate:
    def __init__(self):
        self.choices = [
            ByPass(),
            Rotation(degree=90),
            Rotation(degree=180),
            Rotation(degree=270)
        ]

    def __call__(self, image, mask):
        transform = random.choice(self.choices)
        image = transform(image)
        mask = fn.to_pil_image(mask)
        mask = transform(mask)
        mask = fn.to_tensor(mask)
        return image, mask


class RandomCrop:
    def __init__(self, size=256):
        self.size = size, size

    def __call__(self, image, mask):
        y, x, height, width = tr.RandomCrop.get_params(image, self.size)
        patch = fn.crop(image, y, x, height, width)
        label = mask[y:y+height, x:x+width][::2, ::2]
        return patch, label


class MoNuSegTransform:
    def __init__(self):
        self.crop = RandomCrop()
        self.rot = RandomRotate()
        self.norm = tr.Normalize(mean=[0.7198306, 0.43624926, 0.5741127], std=[0.13865258, 0.18120667, 0.14840752])

    def __call__(self, image, mask):
        patch, label = self.crop(image, mask)
        #patch, label = self.rot(patch, label)
        patch = self.norm(fn.to_tensor(patch))

        return patch, torch.from_numpy(label).long()



class MoNuSeg(Dataset):
    def __init__(self, pth_file: str, multiplier=16,
                 training=True, same_organ_testing=False, different_organ_testing=False):
        self.pth_file = pth_file
        error_message = 'choose one of training | same_organ_testing | different_organ_testing, no less, no more.'
        if int(training) + int(same_organ_testing) + int(different_organ_testing) != 1:
            raise ValueError(error_message)
        dataset = torch.load(self.pth_file)
        by_patient_id = dataset['by_patient_id']
        by_organ = dataset['by_organ']

        if training:
            pid = by_organ['Breast'][:4] + by_organ['Liver'][:4] + by_organ['Kidney'][:4] + by_organ['Prostate'][:4]
        elif same_organ_testing:
            pid = by_organ['Breast'][4:] + by_organ['Liver'][4:] + by_organ['Kidney'][4:] + by_organ['Prostate'][4:]
        elif different_organ_testing:
            pid = by_organ['Bladder'] + by_organ['Colon'] + by_organ['Stomach']
        else:
            raise ValueError(error_message)

        self.dataset = [by_patient_id[p][:2] for p in pid]
        self.is_training = training
        if self.is_training:
            self.transform = MoNuSegTransform()
        self.multiplier = multiplier

    def __len__(self):
        return len(self.dataset) * self.multiplier

    def __getitem__(self, idx):
        image, mask = self.dataset[idx % len(self.dataset)]
        return self.transform(image, mask)
