import torch
from torch import nn


class CIBHashLoss(nn.Module):
    def __init__(self, temperature=0.3, beta=0.001, **kwargs):
        super(CIBHashLoss, self).__init__()
        self.losses = {}
        self.temperature = temperature
        self.beta = beta
        self.criterion = NtXentLoss(self.temperature)

    def forward(self, prob_i, prob_j, z_i, z_j):
        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j)
        self.losses['kl'] = kl_loss
        self.losses['contrast'] = contra_loss
        loss = contra_loss + self.beta * kl_loss

        return loss

    def compute_kl(self, prob, prob_v):
        prob_v = prob_v.detach()
        # prob = prob.detach()

        kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (
                torch.log(1 - prob + 1e-8) - torch.log(1 - prob_v + 1e-8))
        kl = torch.mean(torch.sum(kl, axis=1))
        return kl


class NtXentLoss(nn.Module):
    def __init__(self, temperature):
        super(NtXentLoss, self).__init__()
        # self.batch_size = batch_size
        self.temperature = temperature

        # self.mask = self.mask_correlated_samples(batch_size)
        self.similarityF = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).to(z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
