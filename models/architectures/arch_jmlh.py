import torch
import torch.nn as nn

from models import register_network, BaseArch
from models.architectures.helper import get_backbone, get_hash_activation


@register_network('jmlh')
class ArchJMLH(BaseArch):
    """Arch JMLH"""

    def __init__(self, config, **kwargs):
        super(ArchJMLH, self).__init__(config, **kwargs)
        self.backbone = get_backbone(backbone=self.backbone_name, nbit=self.nbit, nclass=self.nclass,
                                     pretrained=self.pretrained, freeze_weight=self.freeze_weight, **kwargs)
        self.hash_fc = nn.Linear(self.backbone.in_features, self.nbit, bias=kwargs.get('bias', True))
        self.ce_fc = nn.Linear(self.nbit, self.nclass)
        self.stochastic_eps = config['loss_param'].get('stochastic_eps', True)
        hash_layer = config['loss_param'].get('hash_layer', 'stochasticbin')
        self.hash_layer = get_hash_activation(hash_layer)
        self.nbit = self.nbit

        nn.init.normal_(self.hash_fc.weight, std=0.01)
        # nn.init.zeros_(self.hash_fc.bias)

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def get_eps(self, x):
        if self.training and self.stochastic_eps:
            return torch.rand_like(x)
        else:
            return torch.ones_like(x) * 0.5

    def forward(self, x):
        x = self.backbone(x)
        code_logits = self.hash_fc(x)
        code, prob = self.hash_layer(code_logits, self.get_eps(code_logits))
        logits = self.ce_fc(code)
        return logits, torch.sigmoid(code_logits), code
