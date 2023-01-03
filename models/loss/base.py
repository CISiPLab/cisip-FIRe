import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BaseLoss, self).__init__()

        self.losses = {}

    def forward(self, *args, **kwargs):
        return torch.tensor(0.)
