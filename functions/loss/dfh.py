import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFHLoss(nn.Module):
    """https://github.com/swuxyj/DeepHash-pytorch/blob/master/DFH.py
    """
    def __init__(self, train_size, nbit, nclass, mu=1, vul=1, m=1, nta=1, eta=0.5, multiclass=False, **kwargs):
        super(DFHLoss, self).__init__()

        self.multiclass = multiclass
        self.mu = mu
        self.vul = vul
        self.m = m
        self.nta = nta
        self.eta = eta

        self.U = torch.zeros(nbit, train_size).float()
        self.Y = torch.zeros(nclass, train_size).float()
        # Relax_center
        self.V = torch.zeros(nbit, nclass)

        # Center
        self.C = self.V.sign()

        T = 2 * torch.eye(self.Y.size(0)) - torch.ones(self.Y.size(0))
        TK = self.V.size(0) * T
        self.TK = torch.FloatTensor(torch.autograd.Variable(TK, requires_grad=False))

        self.losses = {}

        self.centroids = None

    def forward(self, u, y, ind, onehot=True):
        """
        u: codes from backbone
        y: label (onehot)
        ind: index
        """
        assert len(y.shape) == 2, 'Only support one hot yet now.'
        assert ind is not None, 'ind cannot be None'
        y = y.float()
        if self.U.get_device() != u.get_device():
            self.U = self.U.to(u.get_device())
            self.Y = self.Y.to(u.get_device())
            self.C = self.C.to(u.get_device())
            self.TK = self.TK.to(u.get_device())
            self.V = self.V.to(u.get_device())
        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        b = (self.mu * self.C @ y.t() + u.t()).sign()

        # self.center_gradient(torch.autograd.Variable(self.V, requires_grad=True),
        #                      torch.autograd.Variable(y, requires_grad=False),
        #                      torch.autograd.Variable(b, requires_grad=False))

        self.discrete_center(torch.autograd.Variable(self.C.t(), requires_grad=True),
                             torch.autograd.Variable(y.t(), requires_grad=False),
                             torch.autograd.Variable(b, requires_grad=False))

        s = (y @ self.Y > 0).float()
        inner_product = u @ self.U * 0.5
        inner_product = inner_product.clamp(min=-100, max=50)
        metric_loss = ((1 - s) * torch.log(1 + torch.exp(self.m + inner_product))
                       + s * torch.log(1 + torch.exp(self.m - inner_product))).mean()
        # metric_loss = (torch.log(1 + torch.exp(inner_product)) - s * inner_product).mean()  # Without Margin
        quantization_loss = (b - u.t()).pow(2).mean()
        self.losses['metric'] = metric_loss
        self.losses['quant'] = quantization_loss
        loss = metric_loss + self.eta * quantization_loss
        return loss

    def center_gradient(self, V, batchy, batchb):
        alpha = 0.03
        for i in range(200):
            intra_loss = (V @ batchy.t() - batchb).pow(2).mean()
            inter_loss = (V.t() @ V - self.TK).pow(2).mean()
            quantization_loss = (V - V.sign()).pow(2).mean()

            loss = intra_loss + self.vul * inter_loss + self.nta * quantization_loss
            self.losses['intra'] = intra_loss
            self.losses['inter'] = inter_loss
            self.losses['quant_center'] = quantization_loss
            loss.backward()

            if i in (149, 179):
                alpha = alpha * 0.1

            V.data = V.data - alpha * V.grad.data

            V.grad.data.zero_()
        self.V = V
        self.C = self.V.sign()

    def discrete_center(self, C, Y, B):
        """
        Solve DCC(Discrete Cyclic Coordinate Descent) problem.
        """
        ones_vector = torch.ones([C.size(0) - 1]).to(C.get_device())
        for i in range(C.shape[0]):
            Q = Y @ B.t()
            q = Q[i, :]
            v = Y[i, :]
            Y_prime = torch.cat((Y[:i, :], Y[i+1:, :]))
            C_prime = torch.cat((C[:i, :], C[i+1:, :]))
            with torch.no_grad():
                C[i, :] = (q - C_prime.t() @ Y_prime @ v - self.vul * C_prime.t()@ones_vector).sign()
        self.C = C.t()
