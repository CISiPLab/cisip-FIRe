import torch
import torch.nn as nn


class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, dim=2):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.dim = dim

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.weight.data.fill_(1.)
        self.bias.data.zero_()

    def forward(self, inp):
        if self.dim == 4:
            gamma = self.weight.view(1, self.num_features, 1, 1)
            beta = self.bias.view(1, self.num_features, 1, 1)
        else:
            gamma = self.weight.view(1, self.num_features)
            beta = self.bias.view(1, self.num_features)

        if self.training:
            if self.dim == 4:
                avg = inp.mean(dim=(0, 2, 3)).view(1, self.num_features, 1, 1)
            else:
                avg = inp.mean(dim=0).view(1, self.num_features)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * avg.view(-1).detach()
        else:
            if self.dim == 4:
                avg = self.running_mean.view(1, self.num_features, 1, 1)
            else:
                avg = self.running_mean.view(1, self.num_features)

        output = inp - avg
        output = output * gamma + beta

        return output
