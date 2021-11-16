import torch.nn.functional as F

from functions.loss.base_cls import BaseClassificationLoss


class CELoss(BaseClassificationLoss):
    def __init__(self, ce=1, multiclass=False, **kwargs):
        super(CELoss, self).__init__()

        self.ce = ce
        self.multiclass = multiclass

    def forward(self, logits, code_logits, labels, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()
            # loss_ce = F.binary_cross_entropy_with_logits(logits, labels)
            log_logits = F.log_softmax(logits, dim=1)
            labels_scaled = labels / labels.sum(dim=1, keepdim=True)
            loss_ce = - (labels_scaled * log_logits).sum(dim=1)
            loss_ce = loss_ce.mean()
        else:
            if onehot:
                labels = labels.argmax(-1)

            loss_ce = F.cross_entropy(logits, labels)

        self.losses['ce'] = loss_ce

        loss = self.ce * loss_ce
        return loss
