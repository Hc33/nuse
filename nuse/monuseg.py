# encoding: UTF-8

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
import torchvision.transforms.functional as fn

MoNuSeg_MEAN = [0.80994445, 0.59934306, 0.72003025]
MoNuSeg_STD = [0.16179444, 0.21052812, 0.15706065]
MoNUSeg_TEST_ORGANS = ['Breast', 'Liver', 'Kidney', 'Prostate', 'Bladder', 'Colon', 'Stomach']


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


class RandomMirror:
    def __call__(self, image, label):
        destiny = random.random()
        if 0 <= destiny < 1 / 3:
            # left-right
            image = fn.hflip(image)
            label = np.flip(label, (2,))
        if 1 / 3 <= destiny < 2 / 3:
            # top-bottom
            image = fn.vflip(image)
            label = np.flip(label, (1,))
        return image, label


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
        self.mirror = RandomMirror()
        self.norm = tr.Normalize(mean=MoNuSeg_MEAN, std=MoNuSeg_STD)

    def __call__(self, image, label):
        image, label = self.rot(image, label)
        image, label = self.mirror(image, label)
        image = self.norm(fn.to_tensor(image))

        return image, torch.from_numpy(label > 0).float()


class MoNuSegTestTransform:
    def __init__(self):
        self.norm = tr.Normalize(mean=MoNuSeg_MEAN, std=MoNuSeg_STD)
        self.pad = tr.Pad(12, padding_mode='reflect')

    def __call__(self, image, label):
        image = self.norm(fn.to_tensor(self.pad(image)))
        label = torch.from_numpy(label > 0).float()
        return image, label


class MoNuSeg(Dataset):
    def __init__(self, pth_file: str, size=256, stride=248, training=False, testing=False, image_size=1000):
        self.pth_file = pth_file
        error_message = 'choose one of {training, testing}, no less, no more.'
        if int(training) + int(testing) != 1:
            raise ValueError(error_message)
        dataset = torch.load(self.pth_file)
        by_patient_id = dataset['by_patient_id']
        by_organ = dataset['by_organ']

        if training:
            pid = by_organ['Breast'][:4] + by_organ['Liver'][:4] + by_organ['Kidney'][:4] + by_organ['Prostate'][:4]
        elif testing:
            pid = by_organ['Breast'][4:] + by_organ['Liver'][4:] + by_organ['Kidney'][4:] + by_organ['Prostate'][4:] + \
                  by_organ['Bladder'] + by_organ['Colon'] + by_organ['Stomach']
        else:
            raise ValueError(error_message)

        self.dataset = [by_patient_id[p][:3] for p in pid]
        self.is_training = training
        if self.is_training:
            self.transform = MoNuSegTransform()
            self.crop_size = size
            self.stride = stride
            self.step = 1 + (image_size - size) // stride
            self.num_crops = self.step * self.step
        else:
            self.transform = MoNuSegTestTransform()
            self.crop_size, self.stride = image_size, image_size
            self.step, self.num_crops = 1, 1

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
        label = label[:, y:y + self.crop_size, x:y + self.crop_size]
        return self.transform(image, label)

    def test_sample(self, idx):
        image, label, _ = self.dataset[idx % len(self.dataset)]
        return self.transform(image, label)

    def get_regions(self, idx):
        return self.dataset[idx % len(self.dataset)][2]


def create_loaders(datapack, batch_size, size, stride):
    train_loader = DataLoader(MoNuSeg(datapack, training=True, size=size, stride=stride),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MoNuSeg(datapack, testing=True), batch_size=2, shuffle=False)
    return train_loader, test_loader
