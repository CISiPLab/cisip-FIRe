import torch
import torch.nn as nn
from torchvision import models

from models.backbone.base_backbone import BaseBackbone


class ResNetBackbone(BaseBackbone):
    def __init__(self, nbit, nclass, pretrained=False, freeze_weight=False, resnet_size='18', **kwargs):
        super(ResNetBackbone, self).__init__()
        resnet_models = {
            '18': models.resnet18,
            '34': models.resnet34,
            '50': models.resnet50,
            '101': models.resnet101,
            '152': models.resnet152
        }
        model = resnet_models[resnet_size](pretrained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.classifier = model.fc

        resnet_fc_sizes = {
            '18': 512,
            '34': 512,
            '50': 2048,
            '101': 2048,
            '152': 2048
        }
        self.in_features = resnet_fc_sizes[resnet_size]
        self.nbit = nbit
        self.nclass = nclass

        self.params = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4, self.classifier]
        self.params = [list(m.parameters()) for m in self.params]
        self.params = sum(self.params, [])  # join all lists

        if freeze_weight:
            for m in self.modules():
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(False)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_(False)

    def train(self, mode=True):
        super(ResNetBackbone, self).train(mode)
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()

    def get_features_params(self):
        return self.params

    def get_hash_params(self):
        raise NotImplementedError('no hash layer in backbone')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
