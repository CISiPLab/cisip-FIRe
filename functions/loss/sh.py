import logging
import math

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm

import configs


class SHLoss(nn.Module):
    # https://github.com/TreezzZ/SH_PyTorch/blob/master/sh.py

    def __init__(self, nbit, **kwargs):
        super(SHLoss, self).__init__()
        self.pca = None
        self.built = False
        self.nbit = nbit

        self.mean_ = None
        self.components_ = None
        self.whiten = None
        self.explained_variance_ = None

        self.mn = None
        self.R = None
        self.modes = None

        self.setup = False

        self.losses = {}

    def pca_transform(self, x):
        if not self.setup:
            logging.info('Loading PCA for the first time')

            self.mean_ = self.pca.mean_
            self.components_ = self.pca.components_
            self.whiten = self.pca.whiten
            self.explained_variance_ = self.pca.explained_variance_

            if self.mean_ is not None:
                self.mean_ = torch.from_numpy(self.mean_).to(x.device)

            self.components_ = torch.from_numpy(self.components_).to(x.device)
            self.explained_variance_ = torch.from_numpy(self.explained_variance_).to(x.device)

            self.setup = True

        if self.components_.device != x.device:
            if self.mean_ is not None:
                self.mean_ = self.mean_.to(x.device)
            self.components_ = self.components_.to(x.device)
            self.explained_variance_ = self.explained_variance_.to(x.device)

        if self.mean_ is not None:
            x = x - self.mean_.view(1, -1)

        x_pca = x @ self.components_.t()

        if self.whiten:
            x_pca /= torch.sqrt(self.explained_variance_.view(1, -1))

        return x_pca

    def sh_code(self, x):
        x_pca = self.pca_transform(x)
        if self.mn.device != x.device:
            self.mn = self.mn.to(x.device)
            self.R = self.R.to(x.device)
            self.modes = self.modes.to(x.device)
        x_pca = x_pca - self.mn.reshape(1, -1)
        omega0 = math.pi / self.R
        omegas = self.modes * omega0.reshape(1, -1)
        v = torch.zeros(x.size(0), self.nbit).to(x.device)
        for i in range(self.nbit):
            omegai = omegas[i, :]
            ys = torch.sin(x_pca * omegai + math.pi / 2)
            yi = torch.prod(ys, 1)
            v[:, i] = yi
        return v

    def forward(self, x):
        """

        :param x: should be full dataset
        :return:
        """
        if self.training:
            assert not self.built, 'please switch to eval mode'
            device = x.device

            logging.info('Fitting PCA')
            pca = PCA(n_components=self.nbit)
            x_pca = pca.fit_transform(x.cpu().numpy())
            self.pca = pca

            x_pca = torch.from_numpy(x_pca).to(device)

            # Fit uniform distribution
            eps = 1e-7
            mn = x_pca.min(dim=0)[0] - eps
            mx = x_pca.max(dim=0)[0] + eps
            R = mx - mn  # (nbit,)
            # R = the range of every bit

            # Enumerate eigenfunctions
            max_mode = torch.ceil((self.nbit + 1) * R / R.max()).long().cpu()
            n_modes = max_mode.sum().item() - len(max_mode) + 1
            modes = torch.ones(n_modes, self.nbit)
            m = 0
            pbar = tqdm(range(self.nbit), desc='SpecH Train', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                        disable=configs.disable_tqdm)
            for i in pbar:
                modes[m + 1: m + max_mode[i].item(), i] = torch.arange(1, max_mode[i].item()) + 1
                m = m + max_mode[i] - 1

            modes -= 1
            omega0 = math.pi / R
            omegas = modes * omega0.reshape(1, -1).repeat_interleave(n_modes, 0)
            eig_val = -(omegas ** 2).sum(1)
            ii = (-eig_val).argsort()
            modes = modes[ii[1: self.nbit + 1], :]

            self.modes = modes
            self.mn = mn
            self.R = R

            # evaluate
            v = self.sh_code(x)
            self.setup = False

            quan_error = (1 - torch.cosine_similarity(v, v.sign())).mean()
            self.losses['quan'] = quan_error

            self.built = True
            return v, quan_error
        else:
            assert self.built, 'please perform training'
            v = self.sh_code(x)
            return v

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Overrides state_dict() to save also theta value"""
        original_dict = super().state_dict(destination, prefix, keep_vars)
        original_dict['pca'] = self.pca
        original_dict['R'] = self.R
        original_dict['mn'] = self.mn
        original_dict['built'] = self.built
        original_dict['modes'] = self.modes
        return original_dict

    def load_state_dict(self, state_dict, strict=True):
        """ Overrides state_dict() to load also theta value"""
        pca = state_dict.pop('pca')
        built = state_dict.pop('built')
        mn = state_dict.pop('mn')
        R = state_dict.pop('R')
        modes = state_dict.pop('modes')
        self.pca = pca
        self.built = built
        self.R = R
        self.mn = mn
        self.modes = modes
        super().load_state_dict(state_dict, strict)
