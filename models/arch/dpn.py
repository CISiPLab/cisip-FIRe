import torch.nn as nn

from models.arch.base import BaseNet


class DPN(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.hash_fc = nn.Linear(self.backbone.features_size, self.nbit, bias=False)

    def get_training_modules(self):
        return nn.ModuleDict({'hash_fc': self.hash_fc})

    def forward(self, x):
        x = self.backbone(x)
        v = self.hash_fc(x)
        return v
