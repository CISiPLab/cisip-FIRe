import torch
import torch.nn as nn
from torchvision.models import alexnet

from models.backbone.base_backbone import BaseBackbone


class AlexNetBackbone(BaseBackbone):
    def __init__(self, nbit, nclass, pretrained=False, freeze_weight=False, **kwargs):
        super(AlexNetBackbone, self).__init__()

        model = alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        for i in range(6):
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)
        self.classifier = model.classifier[-1]

        self.in_features = model.classifier[6].in_features
        self.nbit = nbit
        self.nclass = nclass

        if freeze_weight:
            for param in self.features.parameters():
                param.requires_grad_(False)
            for param in self.fc.parameters():
                param.requires_grad_(False)

    def get_features_params(self):
        return list(self.features.parameters()) + list(self.fc.parameters()) + list(self.classifier.parameters())

    def get_hash_params(self):
        raise NotImplementedError('no hash layer in backbone')

    def train(self, mode=True):
        super(AlexNetBackbone, self).train(mode)

        # all dropout set to eval
        for mod in self.modules():
            if isinstance(mod, nn.Dropout):
                mod.eval()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
