import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg16_bn

from models.backbone.base_backbone import BaseBackbone


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class VGGBackbone(BaseBackbone):
    def __init__(self, nbit, nclass, pretrained=False, freeze_weight=False, vgg_size='vgg16', **kwargs):
        super(VGGBackbone, self).__init__()
        vgg_sizes = {
            'vgg16': vgg16,
            'vgg16bn': vgg16_bn
        }
        model = vgg_sizes[vgg_size](pretrained)
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = model.classifier[:-1]
        self.classifier = model.classifier[-1]

        self.in_features = 4096
        self.nbit = nbit
        self.nclass = nclass

        if not pretrained:
            _initialize_weights(model)

        if freeze_weight:
            for m in list(self.features) + list(self.fc):
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(False)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_(False)

    def get_features_params(self):
        return list(self.features.parameters()) + list(self.fc.parameters()) + list(self.classifier.parameters())

    def get_hash_params(self):
        raise NotImplementedError('no hash layer in backbone')

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
