import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class PCALoss(nn.Module):
    def __init__(self, nbit, whiten=False, whiten_input=False, **kwargs):
        super(PCALoss, self).__init__()

        # self.R = None
        self.pca = None
        self.whiten = whiten
        self.whiten_input = whiten_input
        self.built = False
        self.nbit = nbit

        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.setup = False

        self.losses = {}

    def whiten_data_points(self, X, method='zca'):
        """
        Whitens the input matrix X using specified whitening method.
        Inputs:
            X:      Input data matrix with data examples along the first dimension
            method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                    'pca_cor', or 'cholesky'.
        """
        X = X.numpy()
        X = X.reshape((-1, np.prod(X.shape[1:])))
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
        W = None

        if method in ['zca', 'pca', 'cholesky']:
            U, Lambda, _ = np.linalg.svd(Sigma)
            if method == 'zca':
                W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
            elif method == 'pca':
                W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
            elif method == 'cholesky':
                W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
        elif method in ['zca_cor', 'pca_cor']:
            V_sqrt = np.diag(np.std(X, axis=0))
            P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
            G, Theta, _ = np.linalg.svd(P)
            if method == 'zca_cor':
                W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
            elif method == 'pca_cor':
                W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
        else:
            raise Exception('Whitening method not found.')

        self.X_mean = nn.Parameter(torch.from_numpy(X_mean).view(1, -1), requires_grad=False)
        self.W = nn.Parameter(torch.from_numpy(W), requires_grad=False)

        return torch.from_numpy(np.dot(X_centered, W.T))

    def forward(self, x):
        """

        :param x: should be full dataset
        :return:
        """
        if self.training:
            assert not self.built, 'please switch to eval mode'
            device = x.device

            if self.whiten_input:
                logging.info('Learning to whiten')
                x = self.whiten_data_points(x.cpu(), 'pca').to(device)

            # R = torch.randn(self.nbit, self.nbit).to(device)
            # [U, _, _] = torch.svd(R)
            # R = U[:, :self.nbit]

            logging.info('Fitting PCA')
            pca = PCA(n_components=self.nbit, whiten=self.whiten)
            x_pca = pca.fit_transform(x.cpu().numpy())
            self.pca = pca

            v = torch.from_numpy(x_pca).to(device)
            v_tilde = v
            quan_error = (1 - torch.cosine_similarity(v_tilde, v_tilde.sign())).mean()

            self.losses['quan'] = quan_error

            # self.R = nn.Parameter(R, requires_grad=False)
            self.built = True
            return v_tilde, quan_error
        else:
            assert self.built, 'please perform training'

            if not self.setup:
                logging.info('Loading PCA for the first time')

                self.mean_ = self.pca.mean_
                self.components_ = self.pca.components_
                self.explained_variance_ = self.pca.explained_variance_

                if self.mean_ is not None:
                    self.mean_ = torch.from_numpy(self.mean_).to(x.device)

                self.components_ = torch.from_numpy(self.components_).to(x.device)
                self.explained_variance_ = torch.from_numpy(self.explained_variance_).to(x.device)

                self.setup = True

                # self.R.data = self.R.data.to(x.device)
                if self.whiten_input:
                    self.X_mean.data = self.X_mean.data.to(x.device)
                    self.W.data = self.W.data.to(x.device)

            if self.whiten_input:
                x_centered = x - self.X_mean
                x = x_centered @ self.W.t()

            if self.mean_ is not None:
                x = x - self.mean_.view(1, -1)

            x_pca = x @ self.components_.t()

            if self.whiten:
                x_pca /= torch.sqrt(self.explained_variance_.view(1, -1))

            # x_pca = self.pca.transform(x.cpu().numpy())
            # x_pca = torch.from_numpy(x_pca).to(x.device)

            v_tilde = x_pca  # @ self.R
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

        if 'R' in state_dict:
            R = state_dict['R']
            self.R = nn.Parameter(R, requires_grad=False)

        if 'W' in state_dict:
            W = state_dict['W']
            X_mean = state_dict['X_mean']

            self.X_mean = nn.Parameter(X_mean, requires_grad=False)
            self.W = nn.Parameter(W, requires_grad=False)

        self.pca = pca
        self.built = built
        super().load_state_dict(state_dict, strict)
