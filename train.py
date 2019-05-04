# encoding: UTF-8

import argparse

import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint

import nuse.logging
from nuse.fcn import FCN
from nuse.loss import criterion
from nuse.monuseg import create_loaders
from nuse.trainer import create_optimizer, create_trainer, create_evaluators, setup_evaluation


def train(args):
    # create model & optimizer
    model = FCN().to(args.device)
    if args.model_state:
        model.load_state_dict(torch.load(args.model_state, 'cpu'))
    optimizer = create_optimizer(args.optimizer, model.parameters(), lr=args.lr)

    # setup dataset & trainers
    train_loader, test_loaders = create_loaders(args.datapack, args.batch_size)
    trainer = create_trainer(args.device, model, optimizer, criterion, clip_grad=args.clip_grad)
    evaluators = create_evaluators(args.device, model, test_loaders, metrics={})

    # setup logging
    nuse.logging.setup_training_logger(trainer, log_filename=args.log_filename, dataset_length=len(train_loader))
    nuse.logging.setup_training_visdom_logger(trainer, model, optimizer, args)
    nuse.logging.setup_testing_logger(evaluators, nuse.monuseg.MoNUSeg_ORGANS_LISTS)

    # setup evaluation during training
    setup_evaluation(trainer, args.evaluate_interval, evaluators, test_loaders)

    # setup checkpoint policy
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              ModelCheckpoint(args.snapshot_dir, args.name,
                                              save_interval=args.snapshot_interval,
                                              n_saved=args.snapshot_max_history,
                                              save_as_state_dict=True,
                                              require_empty=False),
                              {'model': model, 'optimizer': optimizer})

    # launch training
    trainer.run(train_loader, max_epochs=args.max_epochs)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datapack', type=str, default='monuseg.pth', help='path to packed dataset')
    ap.add_argument('--name', type=str, default='nuse', help='name of this run')
    ap.add_argument('--device', type=int, default=0, help='GPU Device ID. Note: it will be convert to `torch.device`')
    ap.add_argument('--model_state', type=str, default=None, help='model state to load')

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

    # setup default visdom environment
    if args.visdom_env is None:
        args.visdom_env = args.name

    # convert GPU ID to torch.device
    if 0 <= args.device < torch.cuda.device_count():
        args.device = torch.device('cuda', args.device)
    else:
        args.device = torch.device('cpu')

    train(args)


if __name__ == '__main__':
    main()
