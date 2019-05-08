# encoding: UTF-8

from dataclasses import dataclass

import numpy as np
import torch
from ignite.engine import Engine, Events
from ignite.engine import _prepare_batch as prepare_batch
from ignite.utils import convert_tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adadelta import Adadelta
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.nn import Module

__all__ = ['MultiLoss', 'Output', 'create_trainer', 'create_evaluator', 'create_optimizer', 'setup_evaluation']


@dataclass
class MultiLoss:
    outside: torch.tensor
    boundary: torch.tensor
    inside: torch.tensor
    overall: torch.tensor

    def backward(self):
        return self.overall.backward()


@dataclass
class Output:
    prediction: torch.tensor
    loss: MultiLoss


@dataclass
class Prediction:
    prediction: torch.tensor
    regions: [np.ndarray]


def create_trainer(device: torch.device, model: torch.nn.Module, optimizer, loss_fn,
                   activation=torch.sigmoid, non_blocking=False, clip_grad=50):
    def on_iteration(engine, batch):
        model.train()
        optimizer.zero_grad()
        if model.factor > 1:
            image, label = batch
            label = label[:, :, ::model.factor, ::model.factor]
            batch = image, label
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        if activation is not None:
            y_pred = activation(y_pred)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if clip_grad is not None:
            clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        return Output(prediction=y_pred, loss=loss)

    return Engine(on_iteration)


def create_evaluator(device: torch.device, model: Module, loader: DataLoader, activation, non_blocking=False):
    region_fn = loader.dataset.get_regions

    def inference(e, x):
        model.eval()
        with torch.no_grad():
            x = convert_tensor(x, device=device, non_blocking=non_blocking)
            batch_size = x.size(0)
            y_pred = model(x)
            if activation is not None:
                y_pred = activation(y_pred)
            regions = [region_fn(i) for i in range(engine.state.index, engine.state.index + batch_size)]
            e.state.index += batch_size
            return Prediction(prediction=y_pred, regions=regions)

    engine = Engine(inference)

    @engine.on(Events.EPOCH_STARTED)
    def init_index(e: Engine):
        e.state.index = 0

    return engine


def create_optimizer(name, parameters, lr):
    if name == 'Adadelta':
        return Adadelta(parameters, lr=lr)
    elif name == 'Adam':
        return Adam(parameters, lr=lr)
    elif name == 'SGD':
        return SGD(parameters, lr=lr)
    else:
        raise KeyError('Unknown optimizer type {!r}. Choose from [Adadelta | Adam | SGD]')


def setup_evaluation(trainer, interval, evaluator, data_loader):
    @trainer.on(Events.EPOCH_COMPLETED)
    def trigger_evaluation(e: Engine):
        if e.state.epoch % interval == 0:
            evaluator.run(data_loader)
