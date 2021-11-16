import torch
import torch.nn as nn


class CustomConv2d(nn.Module):
    def __init__(self, conv, abs=True, selfmod=True):
        super(CustomConv2d, self).__init__()

        self.conv = conv
        self.selfmod = selfmod
        if not selfmod:
            self.scale = nn.Parameter(torch.ones(self.conv.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.conv.out_channels))
        self.avgpool_cache = None
        self.abs = abs

    def get_scale(self, x):
        x = x.mean([2, 3])
        self.avgpool_cache = x

        if self.selfmod:
            return x.unsqueeze(2).unsqueeze(3)
        else:
            return self.scale.view(1, -1, 1, 1)

    def forward(self, x):
        x = self.conv(x)

        if self.abs:
            x = self.get_scale(x).abs() * x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.get_scale(x) * x + self.bias.view(1, -1, 1, 1)

        return x


class CustomConv2dBN(CustomConv2d):
    def __init__(self, conv, bn, abs=True, selfmod=True):
        super(CustomConv2dBN, self).__init__(conv, abs, selfmod)

        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.abs:
            x = self.get_scale(x).abs() * x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.get_scale(x) * x + self.bias.view(1, -1, 1, 1)

        return x
