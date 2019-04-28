# encoding: UTF-8

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


class RandomRotate(tr.RandomChoice):
    def __init__(self):
        super().__init__([
            ByPass(),
            Rotation(degree=90),
            Rotation(degree=180),
            Rotation(degree=270)
        ])


class RandomCrop:
    def __init__(self, size=51):
        self.size = size, size

    def __call__(self, image, mask):
        y, x, height, width = tr.RandomCrop.get_params(image, self.size)
        patch = fn.crop(image, y, x, height, width)
        label = mask[y + height // 2, x + width // 2]
        return patch, label


class MoNuSegTransform:
    def __init__(self):
        self.crop = RandomCrop()
        self.rot = RandomRotate()

    def __call__(self, image, mask):
        patch, label = self.crop(image, mask)
        patch = self.rot(patch)
        return patch, label


class MoNuSeg(Dataset):
    def __init__(self, pth_file: str, sample_per_image=64, iteration_per_epoch=1000,
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
        self.iteration_per_epoch = iteration_per_epoch
        self.sample_per_image = sample_per_image
        if self.is_training:
            self.transform = MoNuSegTransform()

    def __len__(self):
        return len(self.dataset) * self.sample_per_image * self.iteration_per_epoch

    def __getitem__(self, idx):
        batch_image, batch_label = [], []
        for image, mask in self.dataset:
            for _ in range(self.sample_per_image):
                patch, patch_label = self.transform(image, mask)
                batch_image.append(patch)
                batch_label.append(patch_label)
        return torch.stack(batch_image, dim=0), torch.tensor(batch_label).long()
