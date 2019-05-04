# encoding: UTF-8

import argparse

import torch
from ignite.engine import Events, Engine
from ignite.engine import create_supervised_evaluator as create_evaluator
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader

import nuse.logging
from nuse.fcn import FCN
from nuse.loss import criterion
from nuse.monuseg import MoNuSeg
from nuse.trainer import *


def train(args):
    model = FCN().to(args.device)
    if args.model_state:
        model.load_state_dict(torch.load(args.model_state, 'cpu'))
    optimizer = create_optimizer(args.optimizer, model.parameters(), lr=args.lr)
    trainer = create_trainer(args.device, model, optimizer, criterion, clip_grad=args.clip_grad)
    train_loader = DataLoader(MoNuSeg(args.datapack, training=True), batch_size=args.batch_size, shuffle=True)
    evaluator_so = create_evaluator(model, metrics={}, device=args.device)
    evaluator_do = create_evaluator(model, metrics={}, device=args.device)
    so_test_loader = DataLoader(MoNuSeg(args.datapack, same_organ_testing=True), batch_size=2, shuffle=False)
    do_test_loader = DataLoader(MoNuSeg(args.datapack, different_organ_testing=True), batch_size=2, shuffle=False)

    nuse.logging.setup_training_logger(trainer, log_filename=args.log_filename, dataset_length=len(train_loader))
    nuse.logging.setup_training_visdom_logger(trainer, model, optimizer, args)
    nuse.logging.setup_testing_logger(evaluator_so, organs=['Breast', 'Liver', 'Kidney', 'Prostate'])
    nuse.logging.setup_testing_logger(evaluator_do, organs=['Bladder', 'Colon', 'Stomach'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def trigger_evaluation(e: Engine):
        if e.state.epoch % args.evaluate_interval == 0:
            e._logger.info(f'Starting evaluation')
            evaluator_so.run(so_test_loader)
            evaluator_do.run(do_test_loader)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              ModelCheckpoint(args.snapshot_dir, args.name,
                                              save_interval=args.snapshot_interval,
                                              n_saved=args.snapshot_max_history,
                                              save_as_state_dict=True,
                                              require_empty=False),
                              {'model': model, 'optimizer': optimizer})

    trainer.run(train_loader, max_epochs=args.max_epochs)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapack', type=str, default='monuseg.pth')
    ap.add_argument('--name', type=str, default='nuse', help='name this run')
    ap.add_argument('--device', type=int, default=0, help='GPU Device ID')
    ap.add_argument('--model_state', type=str, default=None, help='model state to recover')

    ap.add_argument('--max_epochs', type=int, default=128, help='how many epochs you want')
    ap.add_argument('--evaluate_interval', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1.0)
    ap.add_argument('--optimizer', type=str, default='Adadelta')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--clip_grad', type=float, default=50)

    ap.add_argument('--snapshot_dir', type=str, default='snapshot', help='snapshot file to recover')
    ap.add_argument('--snapshot_interval', type=int, default=16)
    ap.add_argument('--snapshot_max_history', type=int, default=128)

    ap.add_argument('--visdom_server', type=str, default='localhost')
    ap.add_argument('--visdom_port', type=int, default=8097)
    ap.add_argument('--visdom_env', type=str, default=None)

    ap.add_argument('--log_filename', type=str, default='nuse.log')
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
