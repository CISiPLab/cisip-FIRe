import torch.nn as nn


class BaseUnsupervisedLoss(nn.Module):
    def __init__(self):
        super(BaseUnsupervisedLoss, self).__init__()
        self.losses = {}

    def forward(self, x, h, b, labels):
        raise NotImplementedError


class BaseUnsupervisedReconstructionLoss(nn.Module):
    def __init__(self):
        super(BaseUnsupervisedReconstructionLoss, self).__init__()
        self.losses = {}

    def forward(self, x, h, y):
        """

        :param x: feature
        :param h: h = hash(feature) e.g. 4096 -> 64
        :param y: reconstructed feature
        :return:
        """
        raise NotImplementedError
