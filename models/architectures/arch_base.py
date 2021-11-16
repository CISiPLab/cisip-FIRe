import abc
from typing import List

import torch.nn as nn


class BaseArch(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config, **kwargs):
        """Base constructor"""
        super(BaseArch, self).__init__()
        self.backbone_name = config['arch_kwargs']['backbone']
        self.nbit = config['arch_kwargs']['nbit']
        self.nclass = config['arch_kwargs']['nclass']
        self.pretrained = config['arch_kwargs'].get('pretrained', True)
        self.freeze_weight = config['arch_kwargs'].get('freeze_weight', False)
        self.bias = config['arch_kwargs'].get('bias', False)
        self.config = config
        self.hash_kwargs = config['loss_param']

    @abc.abstractmethod
    def get_features_params(self) -> List:
        """Return backbone trainable features"""
        raise NotImplementedError('Please implement this method.')

    @abc.abstractmethod
    def get_hash_params(self) -> List:
        """Return hash layer trainable features"""
        raise NotImplementedError('Please implement this method.')

    @abc.abstractmethod
    def forward(self, x):
        """Implement forward pass of the architecture."""
        raise NotImplementedError('Please implement this method.')
