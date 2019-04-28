# encoding: UTF-8

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from baseline.fcn import FCN
from baseline.monuseg import MoNuSeg


def train(num_epoch, device):
    model = FCN().cuda(device=device)
    model.train()
    optimizer = torch.optim.Adadelta(model.parameters())
    dataset = MoNuSeg('monuseg.pth', training=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(1, num_epoch + 1):
        for iteration, (image, label) in enumerate(loader):
            image = image.to(device)
            hypot = model(image).view(64, 3, -1)
            truth = label.to(device).view(64, -1)

            loss = cross_entropy(hypot, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss = loss.item()
                accuracy = (hypot.max(dim=1)[1] == truth).long().sum().item() / label.numel()

            print(f'Epoch {epoch:4d} Iteration {iteration:4d} loss = {loss:.4f} accuracy = {accuracy:.4f}')
        if epoch % 16:
            torch.save(model.state_dict(), f'snapshot/{epoch}.pth')


def main():
    train(num_epoch=1000, device=0)


if __name__ == '__main__':
    main()
