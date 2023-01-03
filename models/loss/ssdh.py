import logging

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from tqdm import tqdm

import engine


def obtain_codes(model, config, loader):
    train_dataset = hydra.utils.instantiate(config.dataset.train_dataset_without_transform)
    train_loader = engine.dataloader(train_dataset, config.batch_size, shuffle=False, drop_last=False)

    model.eval()

    pbar = tqdm(train_loader, desc='Obtain Codes', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')

    ret_codes = []

    device = next(model.parameters()).device

    for i, (data, labels, index) in enumerate(pbar):
        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)
            feats, codes = model(data)

            ret_codes.append(feats.cpu())

    return torch.cat(ret_codes), device


class SemanticStructureDHLoss(nn.Module):
    # https://github.com/yangerkun/IJCAI2018_SSDH/blob/master/main.py.

    def compute_similarity(self, model, config, train_loader):
        """
        equation 2
        """
        train_feats, device = obtain_codes(model, config, train_loader)

        # Calculate cosine distance
        logging.info('Calculate cosine distance')
        train_feats_norm = F.normalize(train_feats, dim=1, p=2)
        euc_dis = 1 - (train_feats_norm @ train_feats_norm.t()).cpu().numpy()
        # euc_ = pdist(train_feats.cpu().numpy(), 'cosine')
        # euc_dis = squareform(euc_)
        orig_euc_dis = euc_dis
        start = -0.00000001
        margin = 1.0 / 100
        num = np.zeros(100)
        max_num = 0.0
        max_value = 0.0

        # Histogram distribution
        logging.info('Histogram distribution')
        pbar = tqdm(range(100), desc='Histogram', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
        for i in pbar:
            end = start + margin
            temp_matrix = (euc_dis > start) & (euc_dis < end)
            num[i] = np.sum(temp_matrix)
            if num[i] > max_num:
                max_num = num[i]
                max_value = start
            start = end
        euc_dis = euc_dis.reshape(-1, 1)

        # left = []
        # right = []
        # pbar = tqdm(range(euc_dis.shape[0]), desc='Separation', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
        # for i in pbar:
        #     if euc_dis[i] <= max_value:
        #         left.append(euc_dis[i])
        #     else:
        #         right.append(euc_dis[i])
        # left = np.array(left)
        # right = np.array(right)

        logging.info('Separation')
        left = euc_dis[euc_dis <= max_value]
        right = euc_dis[euc_dis > max_value]

        fake_left = self.alpha * max_value - right
        fake_right = self.beta * max_value - left
        left_all = np.concatenate([left, fake_right])
        right_all = np.concatenate([fake_left, right])

        # Gaussian distribution approximation
        logging.info('Gaussian distribution approximation')
        l_mean, l_std = norm.fit(left_all)
        r_mean, r_std = norm.fit(right_all)

        # Obtain fake labels
        logging.info('Obtain fake labels')
        S1 = (orig_euc_dis < l_mean - self.alpha * l_std) * 1.0
        S2 = (orig_euc_dis > r_mean + self.beta * r_std) * (-1.0)
        S = S1 + S2
        self.S = torch.from_numpy(S).long()

        logging.info(f'Similarity matrix: {self.S.size()}')

    def __init__(self, alpha=2, beta=2, **kwargs):
        super(SemanticStructureDHLoss, self).__init__()

        self.alpha = alpha  # similar threshold
        self.beta = beta  # dissimilar threshold
        self.S = None

        self.losses = {}

    def forward(self, codes, index):
        assert self.S is not None, 'please run prepare_dataset_from_model before calling this'

        S_batch = self.S[index, :][:, index].to(codes.device)  # (bs, bs)
        H = codes @ codes.t()  # (bs, bs)
        H_norm = H / codes.size(1)
        loss = S_batch.abs() * torch.pow(H_norm - S_batch, 2)
        loss = loss.mean()

        self.losses['loss'] = loss

        return loss
