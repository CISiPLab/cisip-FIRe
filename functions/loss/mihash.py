import torch
import torch.nn as nn

from functions.hashing import get_sim


class MIHashLoss(nn.Module):
    def __init__(self,
                 nbins=16,
                 train_size=0, nbit=0, nclass=0, keep_train_size=False, **kwargs):
        """
        beta: the beta scale for continuation
        alpha: the alpha in loss function
        """
        super(MIHashLoss, self).__init__()

        self.nbins = nbins
        self.losses = {}

    def forward(self, u, y, ind=None):
        """
        u: fc output (N * nbit)
        y: onehot label (N * C)
        """
        assert len(y.size()) == 2, 'y is an one-hot vector'
        N, nbits = u.size()

        aff = get_sim(y, y).float()
        aff = aff - torch.diag(torch.diag(aff))
        xp = aff
        xn = 1 - aff

        phi = 2 * torch.sigmoid(u) - 1
        dist = (u.size(1) - phi @ phi.t()) / 2

        prCp = torch.sum(xp, 1) / (N - 1)
        prCn = 1 - prCp

        if self.nbins == 0:
            nbins = nbits // 2
        else:
            nbins = self.nbins

        pDCp = torch.zeros(N, nbins).to(u.get_device())
        pDCn = torch.zeros(N, nbins).to(u.get_device())

        delta = nbits // nbins
        centers = torch.arange(0, nbits, delta)

        for b in range(nbins):
            mid = centers[b].item()
            ind = (mid - delta < dist) & (dist <= mid + delta)
            y = 1 - torch.abs(dist - mid) / delta
            pulse = y * ind.float()
            pDCp[:, b] = torch.sum(pulse * xp, 1)
            pDCn[:, b] = torch.sum(pulse * xn, 1)

        pD = (pDCp + pDCn) / (N - 1)

        # normalization
        sum_pDCp = torch.sum(pDCp, 1)
        nz_p = sum_pDCp > 0
        sum_pDCn = torch.sum(pDCn, 1)
        nz_n = sum_pDCn > 0

        pDCp[nz_p, :] /= sum_pDCp[nz_p].unsqueeze(1)
        pDCn[nz_n, :] /= sum_pDCn[nz_n].unsqueeze(1)

        # entropy
        def ent(p):
            # logp = torch.zeros_like(p)
            # logp[p > 0] = torch.log(p[p > 0] + 1e-7)
            logp = torch.log(p + 1e-7)
            return - torch.sum(p * logp, 1)

        ent_D = ent(pD)
        ent_D_C = prCp * ent(pDCp) + prCn * ent(pDCn)

        loss = ent_D - ent_D_C

        self.losses['ent_D'] = ent_D.sum()
        self.losses['ent_D_C'] = ent_D_C.sum()

        return loss.sum()
