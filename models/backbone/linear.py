import torch.nn as nn

from models.backbone.base_backbone import BaseBackbone


class LinearBackbone(BaseBackbone):
    def __init__(self, nclass, nbit, in_channels=2048, **kwargs):
        super(LinearBackbone, self).__init__()

        self.in_features = in_channels
        self.nbit = nbit
        self.nclass = nclass

    def get_features_params(self):
        return list()

    def get_hash_params(self):
        raise NotImplementedError('no hash layer in backbone')

    def forward(self, x):
        return x
