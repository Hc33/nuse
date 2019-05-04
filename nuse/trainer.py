# encoding: UTF-8

from dataclasses import dataclass

import torch
from ignite.engine import Engine
from ignite.engine import _prepare_batch as prepare_batch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.adadelta import Adadelta
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


__all__ = ['MultiLoss', 'Output', 'create_trainer', 'create_optimizer']


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


def create_trainer(device: torch.device, model: torch.nn.Module, optimizer, loss_fn, non_blocking=False, clip_grad=50):
    def on_iteration(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if clip_grad is not None:
            clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        return Output(prediction=y_pred, loss=loss)

    return Engine(on_iteration)


def create_optimizer(name, parameters, lr):
    if name == 'Adadelta':
        return Adadelta(parameters, lr=lr)
    elif name == 'Adam':
        return Adam(parameters, lr=lr)
    elif name == 'SGD':
        return SGD(parameters, lr=lr)
    else:
        raise KeyError('Unknown optimizer type {!r}. Choose from [Adadelta | Adam | SGD]')
