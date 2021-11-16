import torch
import torch.nn as nn
import torch.nn.functional as F


def kld_loss(q, p=torch.tensor(0.5)):
    loss = (q * torch.log(q + 1e-7) - q * torch.log(p + 1e-7) +
            (1 - q) * torch.log(1 - q + 1e-7) - (1 - q) * torch.log(1 - p + 1e-7))
    return loss.sum(dim=1).mean()


class JMLH(nn.Module):
    def __init__(self,
                 lmbd=0.1,
                 multiclass=False, **kwargs):
        super(JMLH, self).__init__()

        self.lmbd = lmbd
        self.multiclass = multiclass

        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        """
        logits: (N, class)
        code_logits: (N, nbit)
        labels: labels
        onehot: whether labels is onehot
        """
        # code_qb = torch.sigmoid(code_logits)
        code_qb = code_logits
        kl = kld_loss(code_qb, torch.ones_like(code_qb) * 0.5)
        if self.multiclass:
            log_logits = F.log_softmax(logits, dim=1)
            labels_scaled = labels / labels.sum(dim=1, keepdim=True)
            ce = - (labels_scaled * log_logits).sum(dim=1)
            ce = ce.mean()
            # ce = F.binary_cross_entropy(torch.sigmoid(logits), labels.float())
        else:
            if onehot:
                labels = labels.argmax(1)
            ce = F.cross_entropy(logits, labels)
        loss = kl * self.lmbd + ce

        self.losses['ce'] = ce
        self.losses['kl'] = kl

        return loss
