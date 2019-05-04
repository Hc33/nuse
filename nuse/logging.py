# encoding: UTF-8

import ignite.contrib.handlers.visdom_logger as vl
from ignite.engine import Engine, Events
from nuse.monuseg import Unnormalize, MoNuSeg_STD, MoNuSeg_MEAN
import logging


def _get_loss(output):
    loss = output.loss
    return dict(overall=loss.overall.item(),
                inside=loss.inside.item(),
                boundary=loss.boundary.item(),
                outside=loss.outside.item())


def setup_training_visdom_logger(trainer, model, optimizer, args):
    logger = vl.VisdomLogger(args.visdom_server, args.visdom_port, env=args.visdom_env, use_incoming_socket=False)

    logger.attach(trainer, event_name=Events.ITERATION_COMPLETED,
                  log_handler=vl.OutputHandler(tag='loss', output_transform=_get_loss))

    logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                  log_handler=vl.GradsScalarHandler(model.predict))

    logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                  log_handler=vl.OptimizerParamsHandler(optimizer))

    unnormalize = Unnormalize(mean=MoNuSeg_MEAN, std=MoNuSeg_STD, inplace=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def visualize_first_sample(e: Engine):
        image, label = e.state.batch
        image = unnormalize(image[:4])
        logger.vis.images(image, win='Image')
        y_boundary = label[:4, 1:2].float()
        logger.vis.images(y_boundary, win='Boundary/y')
        h_boundary = e.state.output.prediction[:4, 1:2]
        logger.vis.images(h_boundary, win='Boundary/h')

    return logger


def setup_training_logger(trainer, log_filename, dataset_length):
    logger = trainer._logger  # type: logging.Logger
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

    @trainer.on(Events.EPOCH_STARTED)
    def log_next_epoch(e: Engine):
        e._logger.info(f'Starting epoch {e.state.epoch:4d} / {e.state.max_epochs:4d}')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(e: Engine):
        epoch, iteration, loss = e.state.epoch, e.state.iteration, e.state.output.loss.overall
        iteration %= dataset_length
        e._logger.info(f'Epoch {epoch:4d} Iteration {iteration:4d} loss = {loss:.4f}')


def setup_testing_logger(evaluators, organ_lists):
    pass
