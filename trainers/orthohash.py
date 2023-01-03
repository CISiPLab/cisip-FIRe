import random

import numpy as np
import torch
from scipy.linalg import hadamard

from trainers.base import BaseTrainer
from utils import io
from utils.hashing import get_hamm_dist
from utils.metrics import calculate_accuracy_hamm_dist, calculate_accuracy


def get_hadamard(nclass, nbit, fast=True):
    """
    copy from CSQ
    """
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


def get_codebook(codebook_method, nclass, nbit):
    assert codebook_method in ['N', 'B', 'H', 'O']

    if codebook_method == 'N':  # normal
        codebook = torch.randn(nclass, nbit)
    elif codebook_method == 'B':  # bernoulli
        prob = torch.ones(nclass, nbit) * 0.5
        codebook = torch.bernoulli(prob) * 2. - 1.
    elif codebook_method == 'H':  # hadamard
        codebook = get_hadamard(nclass, nbit)
    else:  # O: optim
        codebook = optimize_codebook(nclass, nbit)

    return codebook.sign()


def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)


def optimize_codebook(nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.05):
    """
    brute force to find centroid with furthest distance
    :param nclass:
    :param nbit:
    :param maxtries:
    :param initdist:
    :param mindist:
    :param reducedist:
    :return:
    """
    codebook = torch.zeros(nclass, nbit)
    i = 0
    count = 0
    currdist = initdist
    while i < nclass:
        print(i, end='\r')
        c = torch.randn(nbit).sign()
        nobreak = True
        for j in range(i):
            if get_hd(c, codebook[j]) < currdist:
                i -= 1
                nobreak = False
                break
        if nobreak:
            codebook[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            print('reduce', currdist, i)
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1
    codebook = codebook[torch.randperm(nclass)]
    return codebook


class OrthoHashTrainer(BaseTrainer):
    def __init__(self, config):
        super(OrthoHashTrainer, self).__init__(config)

        self.codebook = None

    def load_model(self):
        super(OrthoHashTrainer, self).load_model()
        self.codebook = self.model.codebook

    def save_codebook(self, fn):
        io.fast_save(self.codebook, fn)

    def load_codebook(self, fn):
        self.codebook = torch.load(fn)

    def load_for_inference(self, logdir):
        self.load_codebook(f'{logdir}/outputs/codebook.pth')

    def to_device(self, device=None):
        if device is None:
            device = self.device

        self.model = self.model.to(device)
        self.codebook = self.codebook.to(device)

    def is_ready_for_inference(self):
        ready = super(OrthoHashTrainer, self).is_ready_for_inference()
        ready = ready and self.codebook is not None
        return ready

    def is_ready_for_training(self):
        ready = super(OrthoHashTrainer, self).is_ready_for_training()
        ready = ready and self.codebook is not None
        return ready

    def save_before_training(self, logdir):
        super(OrthoHashTrainer, self).save_before_training(logdir)
        self.save_codebook(f'{logdir}/outputs/codebook.pth')

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args

        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            logits, codes = self.model(image)
            loss = self.criterion(logits, codes, labels)
            acc = calculate_accuracy(logits, labels)

            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))
            meters['acc'].update(acc.item(), image.size(0))
            meters['hacc'].update(hacc.item(), image.size(0))

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        logits, codes = self.model(image)

        loss = self.criterion(logits, codes, labels)

        # backward and update
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = calculate_accuracy(logits, labels)
            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
        meters['acc'].update(acc.item(), image.size(0))
        meters['hacc'].update(hacc.item(), image.size(0))
