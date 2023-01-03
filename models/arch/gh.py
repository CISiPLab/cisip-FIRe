import torch.nn as nn

from models.arch.base import BaseNet
from models.layers.signhash import SignHashLayer


class SupGreedyHash(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.hash_fc = nn.Linear(self.backbone.features_size, self.nbit)
        self.hash_layer = SignHashLayer()
        self.ce_fc = nn.Linear(self.nbit, self.nclass)

    def get_training_modules(self):
        return nn.ModuleDict({'hash_fc': self.hash_fc, 'ce_fc': self.ce_fc})

    def forward(self, x):
        """

        :param x:
        :return:
            u = logits
            h = code logits
        """
        x = self.backbone(x)
        h = self.hash_fc(x)
        b = self.hash_layer(h)
        u = self.ce_fc(b)
        return u, h


class UnsupGreedyHash(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.hash_fc = nn.Linear(self.backbone.features_size, self.nbit)
        self.hash_layer = SignHashLayer()

    def get_training_modules(self):
        return nn.ModuleDict({'hash_fc': self.hash_fc})

    def forward(self, x):
        x = self.backbone(x)
        h = self.hash_fc(x)
        b = self.hash_layer(h)
        return x, h, b
