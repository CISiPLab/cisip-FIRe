import torch
import torch.nn as nn


class DTSHLoss(nn.Module):
    """https://github.com/swuxyj/DeepHash-pytorch/blob/master/DTSH.py
    """
    def __init__(self, train_size, nbit, nclass, alpha=5, lmbd=1, **kwargs):
        super(DTSHLoss, self).__init__()

        self.alpha = alpha
        self.lmbd = lmbd

        self.losses = {}

    def forward(self, u, y, ind, onehot=True):
        """
        u: codes from backbone
        y: label (onehot)
        ind: index
        """
        assert len(y.shape) == 2, 'Only support one hot yet now.'
        y = y.float()

        inner_product = u @ u.t()
        s = y @ y.t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0)
                          - self.alpha).clamp(min=-100, max=50)
                loss1 += -(triple - torch.log(1. + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = torch.tensor(0)

        loss2 = self.lmbd * (u - u.sign()).pow(2).mean()

        self.losses['likelihood'] = loss1
        self.losses['quant'] = loss2
        loss = loss1 + loss2
        return loss
