import torch.nn as nn

from models import register_network, BaseArch
from models.architectures.helper import get_hash_fc_with_normalizations, get_backbone
from models.layers.cossim import CosSim


@register_network('orthohash')
class ArchOrthoHash(BaseArch):
    """Arch OrthoHash"""

    def __init__(self, config, **kwargs):
        super(ArchOrthoHash, self).__init__(config, **kwargs)

        hash_layer = config['loss_param'].get('hash_layer', 'identity')
        hash_kwargs = config['loss_param']
        cossim_ce = config['loss_param'].get('cossim_ce', True)
        learn_cent = config['loss_param'].get('learn_cent', True)

        self.backbone = get_backbone(backbone=self.backbone_name,
                                     nbit=self.nbit,
                                     nclass=self.nclass,
                                     pretrained=self.pretrained,
                                     freeze_weight=self.freeze_weight, **kwargs)

        if cossim_ce:
            self.ce_fc = CosSim(self.nbit, self.nclass, learn_cent)
        else:
            self.ce_fc = nn.Linear(self.nbit, self.nclass)

        self.hash_fc = get_hash_fc_with_normalizations(in_features=self.backbone.in_features,
                                                       nbit=self.nbit,
                                                       bias=self.bias,
                                                       kwargs=hash_kwargs)

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.backbone(x)
        v = self.hash_fc(x)
        u = self.ce_fc(v)
        return u, v
