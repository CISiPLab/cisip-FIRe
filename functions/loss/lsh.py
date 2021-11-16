import torch
import torch.nn as nn


class LSHLoss(nn.Module):
    """
    Locality-Sensitive Hashing
    """

    def __init__(self, nbit, **kwargs):
        super(LSHLoss, self).__init__()
        self.nbit = nbit
        self.W = None
        self.built = False
        self.losses = {}

    def forward(self, x):
        if self.training:
            assert not self.built, 'please switch to eval mode'
            device = x.device

            self.W = nn.Parameter(torch.randn(self.nbit, x.size(1), device=device), requires_grad=False)

            v = x @ self.W.t()
            quan_error = (1 - torch.cosine_similarity(v, v.sign())).mean()

            self.losses['quan'] = quan_error

            self.built = True
            return v, quan_error
        else:
            assert self.built, 'please perform training'

            v = x @ self.W.t()
            return v

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Overrides state_dict() to save also theta value"""
        original_dict = super().state_dict(destination, prefix, keep_vars)
        original_dict['W'] = self.W.state_dict()
        original_dict['built'] = self.built
        return original_dict

    def load_state_dict(self, state_dict, strict=True):
        """ Overrides state_dict() to load also theta value"""
        W = state_dict.pop('W')
        built = state_dict.pop('built')
        self.W = nn.Parameter(W, requires_grad=False)
        self.built = built
        super().load_state_dict(state_dict, strict)
