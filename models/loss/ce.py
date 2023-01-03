import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, multiclass=False, **kwargs):
        super(CELoss, self).__init__()

        self.multiclass = multiclass

        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        """
        logits: for cross entropy
        code_logits: output from hash FC
        labels: labels
        onehot: whether labels is onehot
        """
        if self.multiclass:
            loss = F.binary_cross_entropy(torch.sigmoid(logits), labels.float())
        else:
            if onehot:
                labels = labels.argmax(1)
            loss = F.cross_entropy(logits, labels)

        self.losses['ce'] = loss

        return loss
