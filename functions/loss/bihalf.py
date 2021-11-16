import torch.nn as nn
import torch.nn.functional as F


class BiHalfLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BiHalfLoss, self).__init__()

        self.losses = {}

    def forward(self, x, h, b, labels, index, **kwargs):
        """
        x: features before hash layer
        h: output from hash FC
        b: binary code
        labels: not using (only use to obtain size)
        """
        # case if batch data not even (normally last batch)
        if x.size(0) % 2 != 0:
            labels = labels[:-1]
            b = b[:-1]
            x = x[:-1]
        target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
        target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:]).detach()

        loss = F.mse_loss(target_b, target_x)

        self.losses['mse'] = loss

        return loss
