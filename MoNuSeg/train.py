# encoding: UTF-8

import torch.optim
from torch.nn.functional import softmax, nll_loss
from MoNuSeg.baseline.model import Baseline
from MoNuSeg.baseline.monuseg import MoNuSeg


def train(num_epoch, device):
    model = Baseline().cuda(device=device)
    optimizer = torch.optim.Adadelta(model.parameters())
    dataset = MoNuSeg('monuseg.pth', training=True)
    for epoch in range(1, num_epoch + 1):
        for iteration, (image, label) in enumerate(dataset):
            image = image.to(device)
            label = label.to(device)
            loss = nll_loss(softmax(model(image)), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            print(f'Epoch {epoch:4d} Iteration {iteration:4d} loss = {loss:.4f}')


def main():
    train(num_epoch=100, device=0)


if __name__ == '__main__':
    main()
