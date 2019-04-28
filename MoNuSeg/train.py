# encoding: UTF-8

import torch
import torch.optim
from torch.nn.functional import cross_entropy
from baseline.model import CNN3 
from baseline.monuseg import MoNuSeg


def train(num_epoch, device):
    model = CNN3().cuda(device=device)
    model.train()
    optimizer = torch.optim.Adadelta(model.parameters())
    dataset = MoNuSeg('monuseg.pth', training=True)
    for epoch in range(1, num_epoch + 1):
        for iteration, (image, label) in enumerate(dataset):
            image = image.to(device)
            hypot = model(image)
            truth = label.to(device)

            loss = cross_entropy(hypot, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss = loss.item()
                accuracy = (hypot.max(dim=1)[1] == truth).long().sum().item() / hypot.numel()

            b = (label == 1).long().sum().item() / 1024
            c = (label == 2).long().sum().item() / 1024
            a = 1 - b - c

            print(f'Epoch {epoch:4d} Iteration {iteration:4d} loss = {loss:.4f} accuracy = {accuracy:.4f} {a:.2f} | {b:.2f} | {c:.2f}')
        torch.save(model.state_dict(), f'snapshot/{epoch}.pth')


def main():
    train(num_epoch=100, device=0)


if __name__ == '__main__':
    main()
