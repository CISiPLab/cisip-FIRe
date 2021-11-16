import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.hashing import get_sim, log_trick


class HashNetLoss(nn.Module):
    def __init__(self, beta=1, alpha=1, step_continuation=20, train_size=0, nbit=0, nclass=0,
                 keep_train_size=False, **kwargs):
        """
        reference: https://github.com/swuxyj/DeepHash-pytorch/blob/master/HashNet.py

        beta: the beta scale for continuation
        alpha: the alpha in loss function
        """
        super(HashNetLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.step_continuation = step_continuation
        self.losses = {}

        self.U = None
        self.Y = None

        if keep_train_size:
            self.U = torch.zeros(train_size, nbit)
            self.Y = torch.zeros(train_size, nclass)

        self.label_not_onehot = False
        self.nclass = nclass

    def forward(self, u, y, ind=None):
        """
        u: fc output (N * nbit)
        y: onehot label (N * C)
        """
        if self.label_not_onehot:
            y = F.one_hot(y.long(), self.nclass)
        assert len(y.size()) == 2, 'y must be an one-hot vector'
        y = y.float()

        u = torch.tanh(self.beta * u)

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
            u1 = u
            u2 = u
            y1 = y
            y2 = y

        # s_ij, size = (N, N)
        similarity = get_sim(y1, y2).float()

        # inner product <h_i, h_j>, size = (N, N)
        dot_product = self.alpha * torch.matmul(u1, u2.t())

        # hashnet eq.4:  log(1 + exp(alpha * <h_i, h_j>)) - alpha * s_ij * <h_i, h_j>
        # exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product
        exp_loss = log_trick(dot_product) - similarity * dot_product

        # |S0| and |S1|
        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()

        # |S|
        S = S0 + S1

        # weight: w_ij, c_ij = 1 because s_ij is given,  S / S10 is the imbalance weight
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        # mean in number of samples
        loss = exp_loss.sum() / S
        return loss
