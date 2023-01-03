import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseNet, self).__init__()

        self.features_size = 0

    def forward(self, x):
        raise NotImplementedError('please implement `forward`')

    # def classify(self, x):
    #     raise NotImplementedError('please implement `classify`')
