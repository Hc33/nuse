# encoding: UTF-8

import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as tr
import torchvision.transforms.functional as fn

MoNuSeg_MEAN = [0.7198306, 0.43624926, 0.5741127]
MoNuSeg_STD = [0.13865258, 0.18120667, 0.14840752]


class ByPass:
    def __call__(self, *args):
        return args


class Rotation:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, mask):
        image = fn.rotate(image, self.degree)
        mask = np.rot90(mask, k=self.degree // 90, axes=(1, 2))
        return image, np.ascontiguousarray(mask)


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
        image, mask = transform(image, mask)
        return image, mask


def unnormalize(tensor, mean, std, inplace=True):
    if not inplace:
        tensor = tensor.clone()
    if tensor.dim() == 3:
        for channel, m, s in zip(tensor, mean, std):
            channel.mul_(s).add_(m)
    else:
        for instance in tensor:
            unnormalize(instance, mean, std, inplace)
    return tensor


class Unnormalize:
    def __init__(self, mean, std, inplace=True):
        self.mean, self.std, self.inplace = mean, std, inplace

    def __call__(self, tensor):
        return unnormalize(tensor, self.mean, self.std, self.inplace)


class MoNuSegTransform:
    def __init__(self):
        self.rot = RandomRotate()
        self.norm = tr.Normalize(mean=MoNuSeg_MEAN, std=MoNuSeg_STD)

    def __call__(self, image, label):
        image, label = self.rot(image, label)
        image = self.norm(fn.to_tensor(image))

        return image, torch.from_numpy(label).long()


class MoNuSegTestTransform:
    def __init__(self):
        self.norm = tr.Normalize(mean=MoNuSeg_MEAN, std=MoNuSeg_STD)
        self.pad = tr.Pad(12, padding_mode='reflect')

    def __call__(self, image, mask):
        image = self.norm(fn.to_tensor(self.pad(image)))
        mask = torch.from_numpy(mask)
        return image, mask


class MoNuSeg(Dataset):
    def __init__(self, pth_file: str, size=256, stride=248,
                 training=False, same_organ_testing=False, different_organ_testing=False,
                 image_size=1000):
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

        self.dataset = [by_patient_id[p][:3] for p in pid]
        self.is_training = training
        if self.is_training:
            self.transform = MoNuSegTransform()
        else:
            self.transform = MoNuSegTestTransform()
        self.crop_size = size
        self.stride = stride
        self.step = image_size // stride
        self.num_crops = self.step * self.step

    def __len__(self):
        return len(self.dataset) * self.num_crops

    def __getitem__(self, idx):
        if self.is_training:
            return self.training_sample(idx)
        return self.test_sample(idx)

    def training_sample(self, idx):
        sample_id, crop_id = divmod(idx, self.num_crops)
        step_y, step_x = divmod(crop_id, self.step)
        x = step_x * self.stride
        y = step_x * self.stride
        image, label, _ = self.dataset[sample_id]
        image = fn.crop(image, y, x, self.crop_size, self.crop_size)
        label = label[:, y:y+self.crop_size, x:y+self.crop_size]
        return self.transform(image, label)

    def test_sample(self, idx):
        image, label, _ = self.dataset[idx % len(self.dataset)]
        return self.transform(image, label)

    def get_regions(self, idx):
        return self.dataset[idx % len(self.dataset)][2]
