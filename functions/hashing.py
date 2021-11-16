import random

import numpy as np
import torch
from scipy.linalg import hadamard


def get_hamm_dist(codes, centroids, margin=0, normalize=False):
    with torch.no_grad():
        nbit = centroids.size(1)
        dist = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

        if normalize:
            dist = dist / nbit

        if margin == 0:
            return dist
        else:
            codes_clone = codes.clone()
            codes_clone[codes_clone.abs() < margin] = 0
            dist_margin = 0.5 * (nbit - torch.matmul(codes_clone.sign(), centroids.sign().t()))
            if normalize:
                dist_margin = dist_margin / nbit
            return dist_margin


def get_codes_and_labels(model, loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vs = []
    ts = []
    for e, (d, t) in enumerate(loader):
        print(f'[{e + 1}/{len(loader)}]', end='\r')
        with torch.no_grad():
            # model forward
            d, t = d.to(device), t.to(device)
            v = model(d)
            if isinstance(v, tuple):
                v = v[0]

            vs.append(v)
            ts.append(t)

    print()
    vs = torch.cat(vs)
    ts = torch.cat(ts)
    return vs, ts


def jmlh_dist(a, b):
    # a_m1 = a - 1
    # b_m1 = b - 1
    # c1 = torch.matmul(a, b_m1.T)
    # c2 = torch.matmul(a_m1, b.T)
    # return torch.abs(c1 + c2)

    # a & b is sigmoid input
    a = torch.sign(a - 0.5)
    b = torch.sign(b - 0.5)
    return hamming(a, b)


def hamming(a, b):
    nbit = a.size(1)
    return 0.5 * (nbit - torch.matmul(a, b.t()))  # (Na, nbit) * (nbit, Nb)


def euclidean(a, b):
    # dist = (a.unsqueeze(1) - b.unsqueeze(0)) ** 2
    # dist = dist.sum(dim=-1)
    # dist = dist ** 0.5
    # return dist
    return torch.cdist(a, b, p=2)  # (Na, Nb)


def cosine(a, b):
    a = a / (torch.norm(a, p=2, dim=1, keepdim=True) + 1e-7)
    b = b / (torch.norm(b, p=2, dim=1, keepdim=True) + 1e-7)

    return (1 - torch.matmul(a, b.t())) / 2


def get_distance_func(distance_func):
    if distance_func == 'hamming':
        return hamming
    elif distance_func == 'euclidean':
        return euclidean
    elif distance_func == 'cosine':
        return cosine
    elif distance_func == 'jmlh-dist':
        return jmlh_dist
    else:
        raise ValueError(f'Distance function `{distance_func}` not implemented.')



def inverse_sigmoid(y):
    y = y.clamp(0.0000001, 0.9999999)  # avoid nan
    return torch.log(y / (1 - y))


def sign_dist(inputs, centroids, margin=0):
    n, b1 = inputs.size()
    nclass, b2 = centroids.size()

    assert b1 == b2, 'inputs and centroids must have same number of bit'

    # sl = relu(margin - x*y)
    out = inputs.view(n, 1, b1) * centroids.sign().view(1, nclass, b1)
    out = torch.relu(margin - out)  # (n, nclass, nbit)

    return out


def calculate_similarity_matrix(centroids):
    nclass = centroids.size(0)
    sim = torch.zeros(nclass, nclass, device=centroids.device)

    for rc in range(nclass):
        for cc in range(nclass):
            sim[rc, cc] = (centroids[rc] == centroids[cc]).float().mean()

    return sim


def get_sim(label_a, label_b, onehot=True):
    """
    label_a: (N, 1 or C)
    label_b: (M, 1 or C)

    return: boolean similarity (N, M)
    """
    if onehot:
        sim = torch.matmul(label_a.float(), label_b.float().t())
        return sim >= 1
    else:
        n = label_a.size()
        m = label_b.size()

        label_a = label_a.view(n, 1)
        label_b = label_b.view(1, m)

        sim = label_a == label_b
        return sim


def log_trick(dot_product):
    """
    loss = log(1 + e^(dt)) - s * dt
    """
    return torch.log(1 + torch.exp(-torch.abs(dot_product))) + dot_product.clamp(min=0)


def get_hadamard(nclass, nbit, fast=True):
    H_K = hadamard(nbit)
    H_2K = np.concatenate((H_K, -H_K), 0)
    hash_targets = torch.from_numpy(H_2K[:nclass]).float()

    if H_2K.shape[0] < nclass:
        hash_targets.resize_(nclass, nbit)
        for k in range(20):
            for index in range(H_2K.shape[0], nclass):
                ones = torch.ones(nbit)
                # Bernouli distribution
                sa = random.sample(list(range(nbit)), nbit // 2)
                ones[sa] = -1
                hash_targets[index] = ones

            if fast:
                return hash_targets

            # to find average/min  pairwise distance
            c = []
            # print()
            # print(n_class)
            TF = (hash_targets.view(1, -1, nbit) != hash_targets.view(-1, 1, nbit)).sum(dim=2).float()
            TF_mask = torch.triu(torch.ones_like(TF), 1).bool()
            c = TF[TF_mask]

            # choose min(c) in the range of K/4 to K/3
            # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
            # but it is hard when bit is  small
            if c.min() > nbit / 4 and c.mean() >= nbit / 2:
                print(c.min(), c.mean())
                break

    return hash_targets


