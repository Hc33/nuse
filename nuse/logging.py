# encoding: UTF-8

import ignite.contrib.handlers.visdom_logger as vl
from ignite.engine import Engine, Events
from nuse.monuseg import Unnormalize, MoNuSeg_STD, MoNuSeg_MEAN


def setup_training_visdom_logger(trainer, model, optimizer, args):
    logger = vl.VisdomLogger(args.visdom_server, args.visdom_port, env=args.visdom_env, use_incoming_socket=False)

    logger.attach(trainer, event_name=Events.ITERATION_COMPLETED,
                  log_handler=vl.OutputHandler(tag='loss', output_transform=lambda loss: {'loss': loss}))

    logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                  log_handler=vl.GradsScalarHandler(model))

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


def setup_testing_logger(evaluator, organs):
    pass
