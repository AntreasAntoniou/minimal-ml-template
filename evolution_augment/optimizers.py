import dataclasses

import torch
from transformers import optimization


def get_optimizer(params, config):
    if config.__class__.__name__ == 'SGD':
        config = dataclasses.asdict(config)
        print('Building SGD optimizer with config:\n{}'.format(config))
        return torch.optim.SGD(params=params, **config)
    else:
        raise ValueError('Unsupported optimizer requested')


def get_lr_scheduler(optimizer, config):
    config = dataclasses.asdict(config)
    print('Building LR scheduler with config:\n{}'.format(config))
    return optimization.get_scheduler(optimizer=optimizer, **config)
