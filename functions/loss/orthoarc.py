import torch
import torch.nn.functional as F

from functions.loss.base_cls import BaseClassificationLoss


class OrthoArcLoss(BaseClassificationLoss):
    def __init__(self, ce=1, s=10, m=0.2, multiclass=False, quan=0, quan_type='cs', **kwargs):
        super(OrthoArcLoss, self).__init__()
        self.ce = ce
        self.s = s
        self.m = m
        self.multiclass = multiclass

        self.quan = quan
        self.quan_type = quan_type

    def forward(self, logits, code_logits, labels, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            y_onehot = labels * self.m
            arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
            logits = torch.cos(arc_logits + y_onehot)
            margin_logits = self.s * logits

            # loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
            # imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels, labels.size(1))
            # loss_ce = loss_ce * imbalance_mask
            # loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
            # self.losses['multiclass_acc'] = multiclass_acc

            log_logits = F.log_softmax(margin_logits, dim=1)
            labels_scaled = labels / labels.sum(dim=1, keepdim=True)
            loss_ce = - (labels_scaled * log_logits).sum(dim=1)
            loss_ce = loss_ce.mean()
        else:
            if onehot:
                labels = labels.argmax(1)

            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
            logits = torch.cos(arc_logits + y_onehot)
            margin_logits = self.s * logits

            loss_ce = F.cross_entropy(margin_logits, labels)

        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(code_logits, code_logits.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(code_logits - code_logits.detach().sign())
            else:  # l2
                quantization = torch.pow(code_logits - code_logits.detach().sign(), 2)

            quantization = quantization.mean()
        else:
            quantization = torch.tensor(0.).to(code_logits.device)

        self.losses['ce'] = loss_ce
        self.losses['quantization'] = quantization
        loss = self.ce * loss_ce + self.quan * quantization
        return loss
