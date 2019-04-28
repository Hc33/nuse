# encoding: UTF-8


import torch.nn


class Collapse(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Baseline(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Dropout2d(p=0.1),

            torch.nn.Conv2d(in_channels=3, out_channels=24, kernel_size=9),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.2),

            torch.nn.Conv2d(in_channels=24, out_channels=50, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.25),

            torch.nn.Conv2d(in_channels=50, out_channels=80, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.5),

            Collapse(),

            torch.nn.Linear(in_features=320, out_features=1024),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.Sigmoid(),

            torch.nn.Linear(in_features=1024, out_features=3))


class CNN3(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(in_channels=3, out_channels=25, kernel_size=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=25, out_channels=50, kernel_size=5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=50, out_channels=80, kernel_size=6),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(p=0.25),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            Collapse(),

            torch.nn.Linear(in_features=320, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(in_features=1024, out_features=3))
