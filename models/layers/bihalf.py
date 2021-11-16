import torch
import torch.nn as nn


class BiHalfFunction(torch.autograd.Function):
    gamma = 6
    @staticmethod
    def forward(ctx, u):
        # Yunqiang for half and half (optimal transport) 0.598447
        # _, index = u.sort(0, descending=True)
        # N, D = u.shape
        # B_creat = torch.cat((torch.ones([int(N / 2), D]), -torch.ones([N - int(N / 2), D]))).cuda()
        # b = torch.zeros(u.shape).cuda().scatter_(0, index, B_creat)

        # JT's
        b = (u > u.median(0).values).float() * 2. - 1.

        ctx.save_for_backward(u, b)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = (N, K)
        u, b = ctx.saved_tensors
        lr = BiHalfFunction.gamma / (b.size(0) * b.size(1))  # (1 / NK), gamma=6 from author code
        return grad_output + lr * (u - b)  # (N, K)


class BiHalfLayer(nn.Module):
    def __init__(self, gamma=6):
        super(BiHalfLayer, self).__init__()
        # lazy implementation: global update gamma!
        BiHalfFunction.gamma = gamma

    def extra_repr(self) -> str:
        return f'gamma={BiHalfFunction.gamma}'

    def forward(self, x):
        if self.training:
            return BiHalfFunction.apply(x)
        else:
            return x
