import abc

import torch.nn as nn


class BaseClassificationLoss(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    @abc.abstractmethod
    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError
