import torch.nn as nn

from models.arch.base import BaseNet
from models.layers.bihalf import BiHalfLayer


class BiHalf(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 gamma: float,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.gamma = gamma
        self.hash_fc = nn.Linear(self.backbone.features_size, self.nbit)
        self.hash_layer = BiHalfLayer(self.gamma)

    def get_training_modules(self):
        return nn.ModuleDict({'hash_fc': self.hash_fc})

    def forward(self, x):
        x = self.backbone(x)
        h = self.hash_fc(x)
        b = self.hash_layer(h)
        return x, h, b
