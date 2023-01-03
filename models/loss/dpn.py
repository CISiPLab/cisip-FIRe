import torch
import torch.nn.functional as F

from models.loss.base import BaseLoss


class DPNLoss(BaseLoss):
    def __init__(self, sl=1, margin=1., reg=0.1, multiclass=False, **kwargs):
        super(DPNLoss, self).__init__()

        self.sl = sl
        self.margin = margin
        self.reg = reg
        self.multiclass = multiclass
        self.losses = {}

        self.codebook = None

    def forward(self, code_logits, labels, onehot=True):
        assert self.codebook is not None
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, self.codebook.size(0))
            labels = labels.float()

            bs, nbit = code_logits.size()
            nclass = self.codebook.size(0)

            loss_sl = torch.relu(self.margin - code_logits.view(bs, 1, nbit) * self.codebook.view(1, nclass, nbit))
            loss_sl = loss_sl.sum(2)
            loss_sl = loss_sl * labels.float()
            loss_sl = loss_sl.sum(1).mean()
        else:
            if onehot:
                labels = labels.argmax(1)
            loss_sl = torch.relu(self.margin - code_logits * self.codebook[labels]).sum(1).mean()

        loss_reg = torch.pow(code_logits, 2).mean()

        self.losses['sl'] = loss_sl
        self.losses['reg'] = loss_reg

        loss = self.sl * loss_sl + self.reg * loss_reg
        return loss
