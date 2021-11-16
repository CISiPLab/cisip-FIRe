import torch
import torch.nn as nn
from torch.autograd import Function


class JMLHBinaryActivation(Function):
    @staticmethod
    def forward(ctx, logits, eps):
        prob = 1.0 / (1 + torch.exp(-logits))
        code = (torch.sign(prob - eps) + 1.0) / 2.0
        ctx.save_for_backward(prob)
        return code, prob

    @staticmethod
    def backward(ctx, grad_code, grad_prob):
        prob, = ctx.saved_tensors
        grad_logits = prob * (1 - prob) * (grad_code + grad_prob)

        grad_eps = grad_code
        return grad_logits, grad_eps


class StochasticBinaryLayer(nn.Module):
    def forward(self, x, eps):
        return JMLHBinaryActivation.apply(x, eps)


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


class BinaryActivation(Function):
    # from: https://github.com/doubling/Stochastic_Generative_Hashing/blob/master/sgh_notebook/stochastic_hashing.py
    # from: https://github.com/ymcidence/TBH/blob/master/train/jmlh_train.py

    @staticmethod
    def forward(ctx, input, eps):
        return torch.sign(input - eps).relu()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class BinaryActivationTanh(Function):
    # https://github.com/ssppp/GreedyHash/blob/master/imagenet.py

    @staticmethod
    def forward(ctx, input, eps):
        """
        input: fc output (N, M)
        eps: epsilon for eq.5 and eq.7

        return
        ------
        code: {0, 1}^m
        """
        code = torch.sign(input - eps)
        return code

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class StochasticBinaryActivation(Function):
    # from: https://github.com/Wizaron/binary-stochastic-neurons/blob/master/distributions/distribution.py
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
