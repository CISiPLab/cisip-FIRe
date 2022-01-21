import logging

import faiss
import torch
import torch.nn as nn
from sklearn.manifold import TSNE


def manifold_learning(x, nbit):
    tsne = TSNE(nbit, init='pca', method='exact')
    y = tsne.fit_transform(x)
    return y


class IMHLoss(nn.Module):
    def __init__(self, nbit, kmeans_iters=200, m=400, k=5, bandwidth=512., **kwargs):
        super(IMHLoss, self).__init__()

        self.built = False
        self.nbit = nbit
        self.kmeans_iters = kmeans_iters
        self.m = m  # base set size
        self.k = k  # knn size
        self.bandwidth = bandwidth

        self.kmeans = None
        self.knn_index = None
        self.base_set = None

        self.losses = {}

    def compute_embeddings(self, query):
        """

        :param query: (n, d)
        :param centroids: (m, d)
        :return:
        """
        try:
            query = query.cpu().numpy()
        except:
            pass

        distances, neighbors = self.kmeans.index.search(query, self.k)

        gaussianw = torch.exp(- torch.from_numpy(distances) / self.bandwidth)
        gaussianw = gaussianw / gaussianw.sum(dim=1, keepdim=True)  # (qn, k)

        base_neighbors = self.base_set[neighbors]  # (qn, k, nbit)

        y = (gaussianw.unsqueeze(2) * base_neighbors).sum(dim=1)  # (qn, k, nbit) -> (qn, nbit)
        return y

    def forward(self, x):
        """

        :param x: should be full dataset
        :return:
        """
        if self.training:
            assert not self.built, 'please switch to eval mode'
            device = x.device

            logging.info('Kmeans Learning')
            dim = x.size(1)
            self.kmeans = faiss.Kmeans(d=dim, k=self.m, niter=self.kmeans_iters)
            self.kmeans.train(x.cpu().numpy())

            logging.info('Manifold Learning')
            self.base_set = manifold_learning(self.kmeans.centroids, self.nbit)

            logging.info('Computing Embedding')
            v = self.compute_embeddings(x.cpu().numpy())
            v = v.to(device)

            quan_error = (1 - torch.cosine_similarity(v, v.sign())).mean()

            self.losses['quan'] = quan_error
            self.built = True
            return v, quan_error
        else:
            assert self.built, 'please perform training'

            return self.compute_embeddings(x.cpu().numpy())

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Overrides state_dict() to save also theta value"""
        original_dict = super().state_dict(destination, prefix, keep_vars)
        original_dict['centroids'] = self.kmeans.centroids
        original_dict['base_set'] = self.base_set
        original_dict['built'] = self.built
        original_dict['bandwidth'] = self.bandwidth
        return original_dict

    def load_state_dict(self, state_dict, strict=True):
        """ Overrides state_dict() to load also theta value"""
        centroids = state_dict.pop('centroids')
        base_set = state_dict.pop('base_set')
        built = state_dict.pop('built')
        bandwidth = state_dict.pop('bandwidth')

        dim = centroids.shape[1]
        self.kmeans = faiss.Kmeans(d=dim, k=self.m, niter=self.kmeans_iters)
        self.kmeans.centroids = centroids
        self.kmeans.index = faiss.IndexFlatL2(dim)
        self.kmeans.index.reset()
        self.kmeans.index.add(centroids)

        self.built = built
        self.base_set = base_set
        self.bandwidth = bandwidth
        super().load_state_dict(state_dict, strict)
