import torch
from torch import nn

from models.arch.base import BaseNet
from models.layers.signhash import SignHashLayer


class CIBHash(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.encoder = nn.Sequential(nn.Linear(self.backbone.features_size, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, self.nbit))
        self.hash_layer = SignHashLayer()

    def get_training_modules(self):
        return nn.ModuleDict({'encoder': self.encoder})

    def forward(self, x):
        x = self.backbone(x)
        z = self.encoder(x)
        prob = torch.sigmoid(z)

        if self.training:
            z = prob - 0.5
            z = self.hash_layer(z)

        return prob, z
