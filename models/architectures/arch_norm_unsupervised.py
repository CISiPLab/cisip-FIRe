import torch.nn as nn

from models import register_network, BaseArch
from models.architectures.helper import get_hash_fc_with_normalizations, get_backbone, get_hash_activation


@register_network('norm-unsupervised')
class ArchNormUnsupervised(BaseArch):
    """Arch Normalized Unsupervised: BiHalf, BN, ZM, Unsupervised Greedy Hash"""

    def __init__(self, config, **kwargs):
        super(ArchNormUnsupervised, self).__init__(config, **kwargs)

        hash_layer = config['loss_param'].get('hash_layer', 'identity')
        hash_kwargs = config['loss_param']

        self.backbone = get_backbone(backbone=self.backbone_name,
                                     nbit=self.nbit,
                                     nclass=self.nclass,
                                     pretrained=self.pretrained,
                                     freeze_weight=self.freeze_weight,
                                     **kwargs)
        self.hash_fc = get_hash_fc_with_normalizations(in_features=self.backbone.in_features,
                                                       nbit=self.nbit,
                                                       bias=self.bias,
                                                       kwargs=hash_kwargs)
        self.hash_layer = get_hash_activation(hash_layer)

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.backbone(x)
        h = self.hash_fc(x)
        b = self.hash_layer(h)
        return x, h, b
