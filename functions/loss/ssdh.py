import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from tqdm import tqdm

import configs


def obtain_codes(model, config, loader):
    train_dataset = configs.dataset(config,
                                    filename='train.txt',
                                    transform_mode='test',
                                    skip_preprocess=config['dataset_kwargs'].get('train_skip_preprocess', False))
    train_loader = configs.dataloader(train_dataset, config['batch_size'], shuffle=False, drop_last=False)

    model.eval()

    pbar = tqdm(train_loader, desc='Obtain Codes', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)

    ret_codes = []

    device = next(model.parameters()).device

    for i, (data, labels, index) in enumerate(pbar):
        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)

            x, code_logits, b = model(data)[:3]

            ret_codes.append(x.cpu())

    return torch.cat(ret_codes), device


class SemanticStructureDHLoss(nn.Module):
    # https://github.com/yangerkun/IJCAI2018_SSDH/blob/master/main.py.

    def prepare_dataset_from_model(self, model, config, train_loader, test_loader, db_loader):
        train_feats, device = obtain_codes(model, config, train_loader)

        # Calculate cosine distance
        logging.info('Calculate cosine distance')
        euc_ = pdist(train_feats.cpu().numpy(), 'cosine')
        euc_dis = squareform(euc_)
        orig_euc_dis = euc_dis
        start = -0.00000001
        margin = 1.0 / 100
        num = np.zeros(100)
        max_num = 0.0
        max_value = 0.0

        # Histogram distribution
        logging.info('Histogram distribution')
        pbar = tqdm(range(100), desc='Histogram', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                    disable=configs.disable_tqdm)
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
        # pbar = tqdm(range(euc_dis.shape[0]), desc='Separation', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
        # disable=configs.disable_tqdm)
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

    def forward(self, x, h, b, labels, index):
        assert self.S is not None, 'please run prepare_dataset_from_model before calling this'

        h = torch.tanh(h)

        S_batch = self.S[index, :][:, index].to(h.device)  # (bs, bs)
        H = h @ h.t()  # (bs, bs)
        H_norm = H / h.size(1)
        loss = S_batch.abs() * torch.pow(H_norm - S_batch, 2)
        loss = loss.mean()

        self.losses['loss'] = loss

        return loss
