import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, lmdb):
        super().__init__()

        self.lmdb = lmdb

    def forward(self, x):
        return self.lmdb(x)
