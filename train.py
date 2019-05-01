# encoding: UTF-8

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from nuse.fcn import FCN
from nuse.monuseg import MoNuSeg


def universal_test(model, device, loader, organs):
    model.eval()
    hypot_record, label_record = [], []
    for organ, (image, label) in zip(organs, loader):
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)
            hypot = model(image).max(dim=1)[1][:, 6:506, 6:506]
            currect = (hypot == label).sum().item()
            hypot_record.append(currect)
            label_record.append(label.numel())
            print(f'Accuracy of {organ:8s} = {currect / label.numel():.4f}')
    print(f'Overall Accuracy = {sum(hypot_record) / sum(label_record):.4f}')
    model.train()


def same_organ_test(model, device):
    dataset = MoNuSeg('monuseg.pth', training=False, same_organ_testing=True)
    loader = DataLoader(dataset, batch_size=2)
    organs = ['Breast', 'Liver', 'Kidney', 'Prostate']
    universal_test(model, device, loader, organs)


def different_organ_test(model, device):
    dataset = MoNuSeg('monuseg.pth', training=False, different_organ_testing=True)
    loader = DataLoader(dataset, batch_size=2)
    organs = ['Bladder', 'Colon', 'Stomach']
    universal_test(model, device, loader, organs)


def all_test(model, device):
    print('=' * 20, 'SAME ORGAN TEST', '=' * 20)
    same_organ_test(model, device)
    print('=' * 18, 'DIFFERENT ORGAN TEST', '=' * 17)
    different_organ_test(model, device)
    print('=' * 57)


def train(num_epoch, device, recover=None):
    model = FCN().cuda(device=device)
    if recover is not None:
        print('Loading', recover)
        snapshot = torch.load(recover, 'cpu')
        model.load_state_dict(snapshot)
        all_test(model, device)
    model.train()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
    dataset = MoNuSeg('monuseg.pth', training=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(1, num_epoch + 1):
        for iteration, (image, label) in enumerate(loader):
            batch_size = image.size(0)
            image = image.to(device)
            hypot = model(image).view(batch_size, 3, -1)
            truth = label.to(device).view(batch_size, -1)

            loss = cross_entropy(hypot, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss = loss.item()
                accuracy = (hypot.max(dim=1)[1] == truth).long().sum().item() / label.numel()

            print(f'Epoch {epoch:4d} Iteration {iteration:4d} loss = {loss:.4f} accuracy = {accuracy:.4f}')
        if epoch % 16 == 0:
            torch.save(model.state_dict(), f'snapshot/{epoch}.pth')
            all_test(model, device)


def main():
    import sys
    if len(sys.argv) == 2:
        recover = sys.argv[-1]
    else:
        recover = None
    train(num_epoch=1024, device=0, recover=recover)


if __name__ == '__main__':
    main()
