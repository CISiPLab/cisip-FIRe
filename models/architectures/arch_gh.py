import torch.nn as nn

from models import register_network, BaseArch
from models.architectures.helper import get_backbone, get_hash_activation


@register_network('gh')
class ArchGreedyHash(BaseArch):
    """Arch Greedy Hash"""
    def __init__(self, config, **kwargs):
        super(ArchGreedyHash, self).__init__(config, **kwargs)

        hash_layer = config['loss_param'].get('hash_layer', 'signhash')

        self.backbone = get_backbone(backbone=self.backbone_name,
                                     nbit=self.nbit,
                                     nclass=self.nclass,
                                     pretrained=self.pretrained,
                                     freeze_weight=self.freeze_weight, **kwargs)
        self.hash_fc = nn.Linear(self.backbone.in_features, self.nbit, bias=self.bias)
        self.ce_fc = nn.Linear(self.nbit, self.nclass)
        self.hash_layer = get_hash_activation(hash_layer)

        nn.init.normal_(self.hash_fc.weight, std=0.01)

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.backbone(x)
        code_logits = self.hash_fc(x)
        code = self.hash_layer(code_logits)
        logits = self.ce_fc(code)
        return logits, code_logits, code
