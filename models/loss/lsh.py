import logging

import numpy as np
import torch
import torch.nn as nn


class LsHLoss(nn.Module):
    """
    Locality-Sensitive Hashing
    """

    def __init__(self, nbit, whiten_input=False, **kwargs):
        super(LsHLoss, self).__init__()
        self.nbit = nbit
        self.W = None
        self.built = False
        self.whiten_input = whiten_input
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
        R = None

        if method in ['zca', 'pca', 'cholesky']:
            U, Lambda, _ = np.linalg.svd(Sigma)
            if method == 'zca':
                R = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
            elif method == 'pca':
                R = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
            elif method == 'cholesky':
                R = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
        elif method in ['zca_cor', 'pca_cor']:
            V_sqrt = np.diag(np.std(X, axis=0))
            P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
            G, Theta, _ = np.linalg.svd(P)
            if method == 'zca_cor':
                R = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
            elif method == 'pca_cor':
                R = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
        else:
            raise Exception('Whitening method not found.')

        self.X_mean = nn.Parameter(torch.from_numpy(X_mean).view(1, -1), requires_grad=False)
        self.R = nn.Parameter(torch.from_numpy(R), requires_grad=False)

        return torch.from_numpy(np.dot(X_centered, R.T))

    def forward(self, x):
        if self.training:
            assert not self.built, 'please switch to eval mode'
            device = x.device

            if self.whiten_input:
                logging.info('Learning to whiten')
                x = self.whiten_data_points(x.cpu()).to(device)

            self.W = nn.Parameter(torch.randn(self.nbit, x.size(1), device=device), requires_grad=False)

            v = x @ self.W.t()
            quan_error = (1 - torch.cosine_similarity(v, v.sign())).mean()

            self.losses['quan'] = quan_error

            self.built = True
            return v, quan_error
        else:
            assert self.built, 'please perform training'

            if not self.setup:
                device = x.device

                if self.whiten_input:
                    self.X_mean.data = self.X_mean.data.to(device)
                    self.R.data = self.R.data.to(device)

                self.W.data = self.W.data.to(device)
                self.setup = True

            if self.whiten_input:
                x_centered = x - self.X_mean
                x = x_centered @ self.R.t()

            v = x @ self.W.t()
            return v

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Overrides state_dict() to save also theta value"""
        original_dict = super().state_dict(destination, prefix, keep_vars)
        original_dict['W'] = self.W.data
        original_dict['built'] = self.built
        return original_dict

    def load_state_dict(self, state_dict, strict=True):
        """ Overrides state_dict() to load also theta value"""
        W = state_dict['W']
        built = state_dict.pop('built')
        self.W = nn.Parameter(W, requires_grad=False)

        if 'R' in state_dict:
            R = state_dict['R']
            X_mean = state_dict['X_mean']

            self.X_mean = nn.Parameter(X_mean, requires_grad=False)
            self.R = nn.Parameter(R, requires_grad=False)

        self.built = built
        super().load_state_dict(state_dict, strict)
