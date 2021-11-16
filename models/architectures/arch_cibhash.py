from typing import List

import torch
from torch import nn

from models import register_network
from models.architectures.arch_base import BaseArch
from models.architectures.helper import get_hash_activation, get_backbone


@register_network('cibhash')
class ArchCIBHash(BaseArch):
    def __init__(self, config, **kwargs):
        super(ArchCIBHash, self).__init__(config, **kwargs)

        self.backbone = get_backbone(backbone=self.backbone_name, nbit=self.nbit, nclass=self.nclass,
                                     pretrained=self.pretrained, freeze_weight=self.freeze_weight, **kwargs)
        self.encoder = nn.Sequential(nn.Linear(self.backbone.in_features, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, self.nbit))
        self.hash_layer = get_hash_activation(kwargs.get('hash_layer', 'signhash'))

    def get_features_params(self) -> List:
        return self.backbone.get_features_params()

    def get_hash_params(self) -> List:
        return list(self.encoder.parameters())

    def forward(self, x):
        if self.training:
            imgi, imgj = x
            imgi = self.backbone(imgi)
            prob_i = torch.sigmoid(self.encoder(imgi))
            z_i = self.hash_layer(prob_i - 0.5)

            imgj = self.backbone(imgj)
            prob_j = torch.sigmoid(self.encoder(imgj))
            z_j = self.hash_layer(prob_j - 0.5)

            return prob_i, prob_j, z_i, z_j

        else:
            x = self.backbone(x)
            prob = torch.sigmoid(self.encoder(x))
            z = prob - 0.5

            return z
