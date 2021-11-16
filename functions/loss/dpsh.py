import torch
import torch.nn as nn

from functions.hashing import get_sim, log_trick


class DPSHLoss(nn.Module):
    def __init__(self, alpha=1, train_size=0, nbit=0, nclass=0, keep_train_size=False, **kwargs):
        """
        beta: the beta scale for continuation
        alpha: the alpha in loss function
        """
        super(DPSHLoss, self).__init__()

        self.alpha = alpha
        self.train_size = train_size
        self.nbit = nbit
        self.nclass = nclass
        self.losses = {}

        self.U = None
        self.Y = None

        if keep_train_size:
            self.U = torch.zeros(train_size, nbit)
            self.Y = torch.zeros(train_size, nclass)

    def forward(self, u, y, ind=None):
        """
        u: fc output (N * nbit)
        y: onehot label (N * C)
        """
        assert len(y.size()) == 2, 'y is an one-hot vector'

        if ind is not None:
            if self.U.get_device() != u.get_device():
                self.U = self.U.to(u.get_device())
                self.Y = self.Y.to(u.get_device())

            self.U[ind, :] = u.detach().clone()
            self.Y[ind, :] = y.detach().clone()

        if ind is not None:
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

        # inner product <h_i, h_j>, size = (N, N)
        dot_product = torch.matmul(u1, u2.t()) / 2

        # hashnet eq.4:  log(1 + exp(alpha * <h_i, h_j>)) - alpha * s_ij * <h_i, h_j>
        # likelihood = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product
        likelihood = log_trick(dot_product) - similarity * dot_product

        # if ind is not None:
        #     # data_mask = torch.triu(torch.ones_like(similarity), 1)
        #     # likelihood = likelihood * data_mask
        #     # likelihood = likelihood.sum() / (data_mask.sum() + 1e-7)
        #     likelihood = likelihood.sum() / u.size(0)  # (u.size(0) * self.train_size)
        # else:
        #     likelihood = likelihood.mean()

        # likelihood = likelihood.sum() / u.size(0)

        ### using hashnet imbalance method ###
        # |S0| and |S1|
        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()

        # |S|
        S = S0 + S1

        likelihood[mask_positive] = likelihood[mask_positive] * (S / S1)
        likelihood[mask_negative] = likelihood[mask_negative] * (S / S0)
        likelihood = likelihood.sum() / S

        quantization = (u - u.sign()).pow(2).mean()
        # quantization = (u - u.sign()).pow(2).sum() / (u.size(0) * self.train_size)

        self.losses['likelihood'] = likelihood
        self.losses['quantization'] = quantization

        loss = likelihood + self.alpha * quantization
        return loss
