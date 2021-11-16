import torch.nn as nn

from models import register_network, BaseArch
from models.architectures.helper import get_backbone


@register_network('dpn')
class ArchDPN(BaseArch):
    """Arch DPN, CSQ"""
    def __init__(self, config, **kwargs):
        super(ArchDPN, self).__init__(config, **kwargs)

        self.bias = config['arch_kwargs'].get('bias', False)
        # hash_layer = config['loss_param'].get('hash_layer', 'identity')
        # hash_kwargs = config['loss_param']

        self.backbone = get_backbone(backbone=self.backbone_name,
                                     nbit=self.nbit,
                                     nclass=self.nclass,
                                     pretrained=self.pretrained,
                                     freeze_weight=self.freeze_weight,
                                     **kwargs)
        self.ce_fc = nn.Linear(self.backbone.in_features, self.nclass)
        self.hash_fc = nn.Linear(self.backbone.in_features, self.nbit, bias=self.bias)

        # nn.init.normal_(self.hash_fc.weight, std=0.01)

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.backbone(x)
        u = self.ce_fc(x)
        v = self.hash_fc(x)
        return u, v
