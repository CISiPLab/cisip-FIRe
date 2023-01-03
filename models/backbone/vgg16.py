import torch
import torch.nn as nn
from torchvision.models import vgg16

from models.backbone.base import BaseNet


class VGG16(BaseNet):
    def __init__(self, pretrained=True, **kwargs):
        super(VGG16, self).__init__()

        model = vgg16(pretrained=pretrained)

        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        lastidx = len(model.classifier) - 1
        for i in range(lastidx):
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)
        # self.classifier = model.classifier[lastidx]

        self.features_size = model.classifier[lastidx].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # def classify(self, x):
    #     x = self.forward(x)
    #     x = self.classifier(x)
    #     return x
