import numpy as np
import torch

from models.loss.base import BaseLoss
from utils.hashing import get_sim


class ADSHLoss(BaseLoss):
    def __init__(self,
                 gamma=200,
                 multiclass=False,
                 **kwargs):
        """
        reference: https://github.com/jiangqy/ADSH-AAAI2018/blob/master/ADSH_pytorch/utils/adsh_loss.py
        """
        super(ADSHLoss, self).__init__()

        self.gamma = gamma
        self.multiclass = multiclass

    def calculate_loss(self, V, U_train, Y, Y_train):
        num_samples = U_train.size(0)
        num_database, nbit = V.size()

        onehot = len(Y.size() == 2)
        similarity = get_sim(Y.cpu(), Y_train.cpu(), onehot).float()
        square_loss = (U_train.dot(V.transpose()) - nbit * similarity).pow(2)
        V_omega = V[:num_samples, :]
        quantization_loss = (U_train - V_omega) ** 2
        loss = (square_loss.sum() + self.gamma * quantization_loss.sum()) / (num_samples * num_database)
        return loss

    def update_codes(self, V, U_train, Y, Y_train):
        num_samples, nbit = U_train.size()
        onehot = len(Y.size()) == 2

        U_train = U_train.numpy()
        U_bar = torch.zeros(V.shape[0], nbit).numpy()
        U_bar[:num_samples] = U_train

        similarity = get_sim(Y_train.cpu(), Y.cpu(), onehot).float() * 2. - 1.  # (M,N)

        Q = -2 * nbit * similarity.cpu().numpy().transpose().dot(U_train) - 2 * self.gamma * U_bar

        for k in range(nbit):
            sel_ind = np.setdiff1d([ii for ii in range(nbit)], k)

            V_ = V[:, sel_ind]
            Uk = U_train[:, k]
            U_ = U_train[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))

    def forward(self, u, db_v, y, db_y, ind):
        """
        u: fc output (N * nbit)
        y: onehot label (N * C)
        """
        # relax u
        u = torch.tanh(u)
        v_omega = db_v[ind]

        # s_ij, size = (N, M)
        # use cpu to compute this step, in case gpu memory OOM
        onehot = len(y.size()) == 2
        similarity = get_sim(y.cpu(), db_y.cpu(), onehot).float().to(u.device) * 2. - 1.

        # inner product <u, V>, size = (N, M)
        dot_product = torch.matmul(u, db_v.t())

        # minimize square loss
        square_loss = (dot_product - u.size(1) * similarity).pow(2)

        # minimize quantization loss
        quan_loss = self.gamma * (v_omega - u).pow(2)

        # mean in number of samples
        loss = (square_loss.sum() + quan_loss.sum()) / (u.size(0) * db_v.size(0))

        self.losses['square'] = square_loss.mean()
        self.losses['quan'] = quan_loss.mean()

        return loss
