import logging

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm

import configs


class ITQLoss(nn.Module):
    def __init__(self, nbit, max_iters=3, **kwargs):
        super(ITQLoss, self).__init__()

        self.R = None
        self.pca = None
        self.built = False
        self.nbit = nbit
        self.max_iters = max_iters

        self.mean_ = None
        self.components_ = None
        self.whiten = None
        self.explained_variance_ = None
        self.setup = False

        self.losses = {}

    def forward(self, x):
        """

        :param x: should be full dataset
        :return:
        """
        if self.training:
            assert not self.built, 'please switch to eval mode'
            device = x.device

            R = torch.randn(self.nbit, self.nbit).to(device)
            [U, _, _] = torch.svd(R)
            R = U[:, :self.nbit]

            logging.info('Fitting PCA')
            pca = PCA(n_components=self.nbit)
            x_pca = pca.fit_transform(x.cpu().numpy())
            self.pca = pca

            logging.info('Rotation Training')
            v = torch.from_numpy(x_pca).to(device)

            # Training
            pbar = tqdm(range(self.max_iters), desc='ITQ Train', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                        disable=configs.disable_tqdm)
            for i in pbar:
                v_tilde = v @ R
                b = v_tilde.sign()
                quan_error = (1 - torch.cosine_similarity(v_tilde, v_tilde.sign())).mean()
                pbar.set_postfix({'quan': quan_error.item()})

                [U, _, VT] = torch.svd(b.t() @ v)
                R = (VT.t() @ U.t())

            v_tilde = v @ R
            quan_error = (1 - torch.cosine_similarity(v_tilde, v_tilde.sign())).mean()

            self.losses['quan'] = quan_error

            self.R = nn.Parameter(R, requires_grad=False)
            self.built = True
            return v_tilde, quan_error
        else:
            assert self.built, 'please perform training'

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

                self.R.data = self.R.data.to(x.device)

            if self.mean_ is not None:
                x = x - self.mean_.view(1, -1)

            x_pca = x @ self.components_.t()

            if self.whiten:
                x_pca /= torch.sqrt(self.explained_variance_.view(1, -1))

            # x_pca = self.pca.transform(x.cpu().numpy())
            # x_pca = torch.from_numpy(x_pca).to(x.device)

            v_tilde = x_pca @ self.R
            return v_tilde

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Overrides state_dict() to save also theta value"""
        original_dict = super().state_dict(destination, prefix, keep_vars)
        original_dict['pca'] = self.pca
        original_dict['built'] = self.built
        return original_dict

    def load_state_dict(self, state_dict, strict=True):
        """ Overrides state_dict() to load also theta value"""
        pca = state_dict.pop('pca')
        built = state_dict.pop('built')
        self.pca = pca
        self.built = built
        super().load_state_dict(state_dict, strict)
