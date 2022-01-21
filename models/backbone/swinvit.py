from typing import List

import timm
import torch

from models.backbone.base_backbone import BaseBackbone


class SwinTransformerBackbone(BaseBackbone):
    def __init__(self, nbit, nclass, vit_name, pretrained=False, freeze_weight=False, **kwargs):
        super(SwinTransformerBackbone, self).__init__()

        model = timm.create_model(vit_name, pretrained=pretrained)

        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.layers = model.layers
        self.norm = model.norm
        self.avgpool = model.avgpool
        self.head = model.head

        self.in_features = model.num_features
        self.nbit = nbit
        self.nclass = nclass

        assert freeze_weight is False, \
            'freeze_weight in backbone deprecated. Use --backbone-lr-scale=0 to freeze backbone'

    def get_features_params(self) -> List:
        return list(self.parameters())

    def get_hash_params(self) -> List:
        raise NotImplementedError('no hash layer in backbone')

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x
