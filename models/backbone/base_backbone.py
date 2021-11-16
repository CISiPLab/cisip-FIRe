import abc
from typing import List

import torch.nn as nn


class BaseBackbone(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Base constructor"""
        super(BaseBackbone, self).__init__()

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