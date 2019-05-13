# encoding: UTF-8

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
import torchvision.transforms.functional as fn
from MoNuSeg import traits


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
    def __init__(self, mean, std):
        self.rot = RandomRotate()
        self.mirror = RandomMirror()
        self.norm = tr.Normalize(mean, std)

    def __call__(self, image, label):
        image, label = self.rot(image, label)
        image, label = self.mirror(image, label)
        image = self.norm(fn.to_tensor(image))

        return image, torch.from_numpy(label > 0).float()


class MoNuSegTestTransform:
    def __init__(self, mean, std):
        self.norm = tr.Normalize(mean, std)
        self.pad = tr.Pad(12, padding_mode='reflect')

    def __call__(self, image):
        image = self.norm(fn.to_tensor(self.pad(image)))
        return image


class MoNuSeg(Dataset):
    def __init__(self, pth_file: str, size=256, stride=248, training=False, testing=False, image_size=1000):
        self.pth_file = pth_file
        error_message = 'choose one of {training, testing}, no less, no more.'
        if int(training) + int(testing) != 1:
            raise ValueError(error_message)
        source = torch.load(self.pth_file)

        if training:
            self.tissues = traits.get_training_tissues()
        elif testing:
            self.tissues = traits.get_testing_tissues()
        else:
            raise ValueError(error_message)

        self.images = traits.tissue_filter(source['images'], self.tissues)
        self.labels = traits.tissue_filter(source['labels'], self.tissues)
        self.annotations = traits.tissue_filter(source['annotations'], self.tissues)
        self.mean = source['mean']
        self.std = source['std']

        self.is_training = training
        if self.is_training:
            self.transform = MoNuSegTransform(self.mean, self.std)
            self.crop_size = size
            self.stride = stride
            self.step = 1 + (image_size - size) // stride
            self.num_crops = self.step * self.step
        else:
            self.transform = MoNuSegTestTransform(self.mean, self.std)
            self.crop_size, self.stride = image_size, image_size
            self.step, self.num_crops = 1, 1

    def __len__(self):
        return len(self.images) * self.num_crops

    def __getitem__(self, idx):
        if self.is_training:
            return self.training_sample(idx)
        return self.test_sample(idx)

    def training_sample(self, idx):
        sample_id, crop_id = divmod(idx, self.num_crops)
        step_y, step_x = divmod(crop_id, self.step)
        x = step_x * self.stride
        y = step_x * self.stride
        image = self.images[sample_id]
        label = self.labels[sample_id]
        image = fn.crop(image, y, x, self.crop_size, self.crop_size)
        label = label[:, y:y + self.crop_size, x:y + self.crop_size]
        return self.transform(image, label)

    def test_sample(self, idx):
        image = self.images[idx % len(self.images)]
        return self.transform(image)

    def get_annotation(self, idx):
        return self.annotations[idx % len(self.annotations)]


def create_loaders(datapack, batch_size, size, stride):
    train_loader = DataLoader(MoNuSeg(datapack, training=True, size=size, stride=stride),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MoNuSeg(datapack, testing=True), batch_size=2, shuffle=False)
    mean, std = train_loader.dataset.mean, train_loader.dataset.std
    return train_loader, test_loader, Unnormalize(mean, std)
