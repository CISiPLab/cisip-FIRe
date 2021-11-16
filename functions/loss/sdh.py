import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.hashing import get_sim


class SDHLoss(nn.Module):
    # Deep Hashing for Compact Binary Codes Learning
    def __init__(self, lmbd0=0.001, lmbd1=1, lmbd2=0.001, alpha=1, **kwargs):
        super(SDHLoss, self).__init__()

        self.lmbd0 = lmbd0  # quantization
        self.lmbd1 = lmbd1  # bit variance
        self.lmbd2 = lmbd2  # orthogonality on projection weights
        # self.lmbd3 = lmbd3  # weight decay
        self.alpha = alpha  # supervised signal

        self.weight = None  # for lmbd2 (J3)
        self.losses = {}

    def forward(self, u, y, ind=None):
        """
        u: fc output (N * nbit)
        y: onehot label (N * C)
        """
        assert self.weight is not None, 'please set weight before compute loss'
        u = torch.tanh(u)

        if ind is not None:
            if self.U.get_device() != u.get_device():
                self.U = self.U.to(u.get_device())
                self.Y = self.Y.to(u.get_device())

            self.U[ind, :] = u.detach().clone()
            self.Y[ind, :] = y.detach().clone()

            u1 = u
            u2 = self.U
            y1 = y
            y2 = self.Y
        else:
            batchsize = u.size(0)
            # u1 = u[:batchsize // 2]
            # u2 = u[batchsize // 2:]
            # y1 = y[:batchsize // 2]
            # y2 = y[batchsize // 2:]
            u1 = u2 = u
            y1 = y2 = y

        # s_ij, size = (N, N)
        similarity = get_sim(y1, y2).float()

        j1 = torch.sum((u - u.sign()) ** 2, 1).mean()
        j2_1 = torch.trace(torch.matmul(u.t(), u))

        hdiff = u1.unsqueeze(0) - u2.unsqueeze(1)  # (N, N, K)
        hdiff = torch.matmul(hdiff.unsqueeze(3), hdiff.unsqueeze(2))  # (N, N, K, 1) * (N, N, 1, K) = (N, N, K, K)

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        posn = mask_positive.float().sum()
        negn = mask_negative.float().sum()

        similarity = similarity.view(u.size(0), u.size(0), 1, 1)
        hpos = (hdiff * similarity).sum(dim=(0, 1)) / (posn + 1e-7)  # (N, N, K, K) -> (K, K)
        hneg = (hdiff * (1 - similarity)).sum(dim=(0, 1)) / (negn + 1e-7)
        h = hneg - hpos

        j2_2 = torch.trace(h)
        j2 = (j2_1 + self.alpha * j2_2) / 2

        w = self.weight  # (out-K, in-d)
        ortho = torch.matmul(w, w.t())  # (K, K)
        j3 = torch.sum((ortho - torch.ones_like(ortho)) ** 2) / 2

        # skipped j4 as it is weight decay

        loss = self.lmbd0 * j1 - self.lmbd1 * j2 + self.lmbd2 * j3
        self.losses['j1'] = j1
        self.losses['j2_1'] = j2_1
        self.losses['j2_2'] = j2_2
        self.losses['j3'] = j3

        return loss


class SDHLossC(nn.Module):
    # Deep Hashing for Compact Binary Codes Learning
    def __init__(self, lmbd0=0.001, lmbd1=1, lmbd2=0.001, alpha=1, **kwargs):
        super(SDHLossC, self).__init__()

        self.lmbd0 = lmbd0  # quantization
        self.lmbd1 = lmbd1  # bit variance
        self.lmbd2 = lmbd2  # orthogonality on projection weights
        # self.lmbd3 = lmbd3  # weight decay
        self.alpha = alpha  # supervised signal

        self.weight = None  # for lmbd2 (J3)
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        """
        u: fc output (N * nbit)
        y: onehot label (N * C)
        """
        if onehot:
            labels = labels.argmax(1)

        assert self.weight is not None, 'please set weight before compute loss'
        u = code_logits

        j1 = torch.pow(u - u.sign(), 2).mean()  # torch.sum((u - u.sign()) ** 2, 1).mean()
        j2_1 = torch.trace(torch.matmul(u.t(), u)) / u.size(1)

        j2_2 = F.cross_entropy(logits, labels)
        j2 = j2_1 / 2

        w = self.weight  # (out-K, in-d)
        ortho = torch.matmul(w, w.t())  # (K, K)
        j3 = torch.pow(ortho - torch.ones_like(ortho), 2).mean()  # torch.sum((ortho - torch.ones_like(ortho)) ** 2) / 2

        # skipped j4 as it is weight decay

        loss = self.lmbd0 * j1 - self.lmbd1 * j2 + self.alpha * j2_2 + self.lmbd2 * j3
        self.losses['j1'] = j1
        self.losses['j2_1'] = j2_1
        self.losses['j2_2'] = j2_2
        self.losses['j3'] = j3

        return loss
