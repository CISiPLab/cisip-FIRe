import torch
import torch.nn as nn

from models import register_network, BaseArch
from models.architectures.helper import get_backbone


@register_network('uhbdnn')
class ArchUHBDNN(BaseArch):
    """Arch UHBDNN"""

    def __init__(self, config, **kwargs):
        super(ArchUHBDNN, self).__init__(config, **kwargs)
        self.backbone = get_backbone(backbone=self.backbone_name, nbit=self.nbit, nclass=self.nclass,
                                     pretrained=self.pretrained, freeze_weight=self.freeze_weight, **kwargs)

        self.hash_fc = nn.Sequential(
            nn.Linear(self.backbone.in_features, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, self.nbit)
        )
        self.rec_fc = nn.Linear(self.nbit, self.backbone.in_features)

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return (list(self.hash_fc.parameters()) +
                list(self.rec_fc.parameters()))

    def forward(self, x):
        x = self.backbone(x)
        code_logits = self.hash_fc(x)
        rec_x = self.rec_fc(code_logits)

        return x, code_logits, rec_x


if __name__ == '__main__':
    torch.manual_seed(1234)
    net = ArchUHBDNN(64, 10, False, False, 'alexnet')
    print(net.training)
    net.train()

    data = torch.randn(1, 3, 224, 224)
    x, bbn, rec_x = net(data)
    print(x)
    print(bbn)
    print(rec_x)

    from functions.loss.uhbdnn import UHBDNNLoss

    criterion = UHBDNNLoss(1, 1, 1, nbit=64)
    criterion.B = torch.ones(1, 64)
    criterion(x, bbn, rec_x, [0])
    print(criterion.losses)
    print(criterion.state_dict())
