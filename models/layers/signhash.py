import torch
import torch.nn as nn
from torch.autograd import Function


class SignHash(Function):
    # from: https://github.com/ssppp/GreedyHash/blob/master/imagenet.py

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SignHashLayer(nn.Module):
    def forward(self, x):
        return SignHash.apply(x)


def sign_hash(x):
    return SignHash.apply(x)
