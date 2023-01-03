import torch
import torch.nn as nn
from torchvision.models import alexnet

from models.backbone.base import BaseNet


class AlexNet(BaseNet):
    def __init__(self, pretrained=True, **kwargs):
        super(AlexNet, self).__init__()

        model = alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        for i in range(6):
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)
        self.classifier = model.classifier[6]

        self.features_size = model.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def classify(self, x):
        x = self.forward(x)
        x = self.classifier(x)
        return x
