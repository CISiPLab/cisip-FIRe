import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)


def get_centroid(nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.01):
    centroid = torch.zeros(nclass, nbit)
    i = 0
    count = 0
    currdist = initdist
    curri = -1
    while i < nclass:
        if curri != i:
            curri = i
            logging.info(f'Doing for class {i}')

        c = torch.randn(nbit).sign().float()
        nobreak = True

        # to compare distance with previous classes
        for j in range(i):
            if get_hd(c, centroid[j]) < currdist:
                i -= 1
                nobreak = False
                break

        if nobreak:
            centroid[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            logging.info(f'Max tried for {i}, reducing distance constraint {currdist}')
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1

    # shuffle the centroid
    centroid = centroid[torch.randperm(nclass)]
    return centroid


class DPNLoss(nn.Module):
    def __init__(self, ce=0, sl=1, margin=1., reg=0.1, multiclass=False, **kwargs):
        super(DPNLoss, self).__init__()

        self.ce = ce
        self.sl = sl
        self.margin = margin
        self.reg = reg
        self.multiclass = multiclass
        self.losses = {}

        self.centroids = None

    def forward(self, logits, code_logits, labels, onehot=True):
        assert self.centroids is not None
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            bs, nbit = code_logits.size()
            nclass = self.centroids.size(0)

            loss_ce = F.binary_cross_entropy(torch.sigmoid(logits), labels.float())

            loss_sl = torch.relu(self.margin - code_logits.view(bs, 1, nbit) * self.centroids.view(1, nclass, nbit))
            loss_sl = loss_sl.sum(2)
            loss_sl = loss_sl * labels.float()
            loss_sl = loss_sl.sum(1).mean()
        else:
            if onehot:
                labels = labels.argmax(1)

            loss_ce = F.cross_entropy(logits, labels)
            loss_sl = torch.relu(self.margin - code_logits * self.centroids[labels]).sum(1).mean()

        loss_reg = torch.pow(code_logits, 2).mean()

        self.losses['sl'] = loss_sl
        self.losses['ce'] = loss_ce
        self.losses['reg'] = loss_reg

        loss = self.sl * loss_sl + self.ce * loss_ce + self.reg * loss_reg
        return loss
