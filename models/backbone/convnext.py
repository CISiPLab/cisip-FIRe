from typing import List

import timm

from models.backbone.base_backbone import BaseBackbone


class ConvNextBackbone(BaseBackbone):
    def __init__(self, nbit, nclass, vit_name, pretrained=False, freeze_weight=False, **kwargs):
        super(ConvNextBackbone, self).__init__()

        model = timm.create_model(vit_name, pretrained=pretrained)

        self.forward_features = model.forward_features
        self.stem = model.stem
        self.stages = model.stages
        self.norm_pre = model.norm_pre
        self.selective_pool = model.head.global_pool
        self.norm = model.head.norm
        self.flatten = model.head.flatten

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
        # x = self.forward_features(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        # x = self.selective_pool(x)
        # x = self.norm(x)
        # x = self.flatten(x)
        return x