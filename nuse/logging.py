# encoding: UTF-8

import argparse
import logging

import ignite.contrib.handlers.visdom_logger as vl
from ignite.engine import Engine, Events

from nuse.monuseg import Unnormalize, MoNuSeg_STD, MoNuSeg_MEAN
from nuse.metrics import AJIMetric


def _get_loss(output):
    loss = output.loss
    return dict(overall=loss.overall.item(),
                inside=loss.inside.item(),
                boundary=loss.boundary.item(),
                outside=loss.outside.item())


def setup_visdom_logger(trainer, evaluator, model, optimizer, args):
    logger = vl.VisdomLogger(args.visdom_server, args.visdom_port, env=args.visdom_env, use_incoming_socket=False)

    logger.attach(trainer, event_name=Events.ITERATION_COMPLETED,
                  log_handler=vl.OutputHandler(tag='loss', output_transform=_get_loss))

    logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                  log_handler=vl.GradsScalarHandler(model.predict))

    logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                  log_handler=vl.OptimizerParamsHandler(optimizer))

    logger.attach(evaluator, event_name=Events.EPOCH_COMPLETED,
                  log_handler=vl.OutputHandler(tag='validation', metric_names=['AJI'], another_engine=trainer))

    unnormalize = Unnormalize(mean=MoNuSeg_MEAN, std=MoNuSeg_STD, inplace=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def visualize_first_sample(e: Engine):
        image, label = e.state.batch
        image = unnormalize(image[:4])
        logger.vis.images(image, win='Image')
        y_boundary = label[:4, 1:2].float()
        logger.vis.images(y_boundary, win='Boundary/y')
        h_boundary = e.state.output.prediction[:4, 1:2]
        # TODO implement these somewhere else
        if 'lovasz' in args.criterion.lower():
            h_boundary = (h_boundary > 0).float()
        logger.vis.images(h_boundary, win='Boundary/h')

    return logger


def create_logger(log_filename):
    logger = logging.getLogger('nuse::logger')  # type: logging.Logger
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(fmt='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    if log_filename is not None:
        writer = logging.FileHandler(log_filename)
        writer.setLevel(logging.INFO)
        writer.setFormatter(fmt)
        logger.addHandler(writer)
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    return logger


def setup_training_logger(trainer: Engine, logger: logging.Logger):
    trainer._logger = logger
    @trainer.on(Events.EPOCH_STARTED)
    def log_next_epoch(e: Engine):
        logger.info(f'Starting epoch {e.state.epoch:4d} / {e.state.max_epochs:4d}')

    @trainer.on(Events.EPOCH_STARTED)
    def init_num_batch(e: Engine):
        if e.state.epoch == 1:
            e.state.num_batch = None

    @trainer.on(Events.EPOCH_COMPLETED)
    def set_num_batch(e: Engine):
        if e.state.epoch == 1:
            e.state.num_batch = e.state.iteration

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(e: Engine):
        epoch, iteration, loss = e.state.epoch, e.state.iteration, e.state.output.loss.overall
        if e.state.num_batch is not None:
            iteration = 1 + (iteration - 1) % e.state.num_batch
        logger.info(f'Epoch {epoch:4d} Iteration {iteration:4d} loss = {loss:.4f}')


def setup_testing_logger(evaluator: (Engine, AJIMetric), logger: logging.Logger, organ_lists: [str]):
    evaluator._logger = logger

    metric_name = 'AJI'

    metric = AJIMetric()
    metric.attach(evaluator, metric_name)

    @evaluator.on(Events.EPOCH_STARTED)
    def log_start_evaluation(e: Engine):
        logger.info(f'Starting Evaluation')

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_aji_by_organ(e: Engine):
        batch_size = e.state.output.prediction.size(0)
        start = batch_size * (e.state.iteration - 1)
        for i in range(start, start + batch_size):
            organ = organ_lists[i // 2]
            aji = metric.aji_for_all[i]
            logger.info(f'Organ = {organ:8s} AJI = {aji:.4f}')

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_mean_aji(e: Engine):
        logger.info(f'Overall Mean AJI = {e.state.metrics[metric_name]}')


def echo_args(logger: logging.Logger, args: argparse.Namespace):
    kv = args.__dict__
    key_width = max(map(len, kv.keys()))
    fmt = '{{:{}s}}: {{!r}}'.format(key_width)
    for key, value in kv.items():
        logger.info(fmt.format(key, value))
