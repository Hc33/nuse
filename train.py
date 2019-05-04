# encoding: UTF-8

import torch
import torch.optim
from torch.utils.data import DataLoader
from nuse.fcn import FCN
import argparse
from ignite.engine import create_supervised_evaluator as create_evaluator
from ignite.engine import create_supervised_trainer as create_trainer
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from nuse.monuseg import MoNuSeg
from nuse.loss import dice
import nuse.logging
from dataclasses import dataclass


@dataclass
class MultiLoss:
    outside: torch.tensor
    boundary: torch.tensor
    inside: torch.tensor
    overall: torch.torch

    def backward(self):
        return self.overall.backward()


@dataclass
class Output:
    prediction: torch.tensor
    loss: MultiLoss

    @staticmethod
    def from_ignite(x, y, y_pred, loss):
        return Output(prediction=y_pred, loss=loss)


def criterion(hypot, label):
    h_outside, h_boundary, h_inside = map(lambda t: t.unsqueeze(1), torch.split(hypot, 1, dim=1))
    y_outside, y_boundary, y_inside = map(lambda t: t.unsqueeze(1), torch.split(label.float(), 1, dim=1))
    loss_outside = dice(h_outside, y_outside)
    loss_boundary = dice(h_boundary, y_boundary)
    loss_inside = dice(h_inside, y_inside)
    loss = loss_outside + loss_boundary + loss_inside
    return MultiLoss(outside=loss_outside, boundary=loss_boundary, inside=loss_inside, overall=loss)


def train(args):
    model = FCN()
    if args.model_state:
        model.load_state_dict(torch.load(args.recover, 'cpu'))
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
    trainer = create_trainer(model, optimizer, criterion, device=args.device, output_transform=Output.from_ignite)
    evaluator_so = create_evaluator(model, metrics={}, device=args.device)
    evaluator_do = create_evaluator(model, metrics={}, device=args.device)
    so_test_loader = DataLoader(MoNuSeg(args.datapack, same_organ_testing=True), batch_size=2, shuffle=False)
    do_test_loader = DataLoader(MoNuSeg(args.datapack, different_organ_testing=True), batch_size=2, shuffle=False)

    nuse.logging.setup_training_visdom_logger(trainer, model, optimizer, args)
    nuse.logging.setup_testing_logger(evaluator_so, organs=['Breast', 'Liver', 'Kidney', 'Prostate'])
    nuse.logging.setup_testing_logger(evaluator_do, organs=['Bladder', 'Colon', 'Stomach'])

    @trainer.on(Events.EPOCH_STARTED)
    def log_next_epoch(e: Engine):
        print(f'Starting epoch {e.state.epoch:4d} / {e.state.max_epochs:4d}')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(e: Engine):
        print(f'Epoch {e.state.epoch:4d} Iteration {e.state.iteration:4d} loss = {e.state.output.loss.overall:.4f}')

    @trainer.on(Events.EPOCH_COMPLETED)
    def trigger_evaluation(e: Engine):
        if e.state.epoch % args.evaluate_interval == 0:
            print(f'Starting evaluation')
            evaluator_so.run(so_test_loader)
            evaluator_do.run(do_test_loader)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              ModelCheckpoint(args.snapshot_dir, args.name,
                                              save_interval=args.snapshot_interval,
                                              n_saved=args.snapshot_max_history,
                                              save_as_state_dict=True,
                                              require_empty=False),
                              {'model': model, 'optimizer': optimizer})

    train_loader = DataLoader(MoNuSeg(args.datapack, training=True), batch_size=32, shuffle=True)
    trainer.run(train_loader, max_epochs=args.max_epochs)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapack', type=str, default='monuseg.pth')
    ap.add_argument('--name', type=str, default='nuse', help='name this run')
    ap.add_argument('--device', type=int, default=0, help='GPU Device ID')
    ap.add_argument('--model_state', type=str, default=None, help='model state to recover')

    ap.add_argument('--max_epochs', type=int, default=128, help='how many epochs you want')
    ap.add_argument('--evaluate_interval', type=int, default=16)

    ap.add_argument('--snapshot_dir', type=str, default='snapshot', help='snapshot file to recover')
    ap.add_argument('--snapshot_interval', type=int, default=16)
    ap.add_argument('--snapshot_max_history', type=int, default=128)

    ap.add_argument('--visdom_server', type=str, default='localhost')
    ap.add_argument('--visdom_port', type=int, default=8097)
    ap.add_argument('--visdom_env', type=str, default=None)
    return ap


def main():
    args = build_argparser().parse_args()
    if args.visdom_env is None:
        args.visdom_env = args.name
    if 0 <= args.device < torch.cuda.device_count():
        args.device = torch.device('cuda', args.device)
    train(args)


if __name__ == '__main__':
    main()
