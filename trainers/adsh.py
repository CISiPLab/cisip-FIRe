import logging
from collections import defaultdict

import torch
from tqdm import tqdm

import engine
from trainers.base import BaseTrainer
from utils.datasets import subset_dataset
from utils.misc import AverageMeter


def solve_dcc(B, U, expand_U, S, nbit, gamma=200):
    """
    Solve DCC problem.
    """
    Q = (nbit * S).t() @ U + gamma * expand_U

    with tqdm(range(nbit), desc='Bit:', bar_format='{l_bar}{bar:10}{r_bar}') as tbits:
        for bit in tbits:
            q = Q[:, bit]
            u = U[:, bit]
            B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
            U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

            B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


class ADSHTrainer(BaseTrainer):
    """
    Asymmetric deep supervised hashing

    https://github.com/jiangqy/ADSH-AAAI2018
    """

    def __init__(self, config, **kwargs):
        super(ADSHTrainer, self).__init__(config)

        self.S = None
        self.Y = None
        self.B = None  # retrieval
        self.U = None  # training
        self.randidxs = None

    def _get_labels(self, dataloader):
        Y = []
        with tqdm(dataloader, desc='Label:', bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            for i, data in enumerate(tepoch):
                image, labels, index = data
                Y.append(labels.argmax(1))
        return torch.cat(Y)

    def prepare_before_first_epoch(self):
        num_train = len(self.dataset['train'])
        nbit = self.config.model.nbit
        num_samples = self.config.method_params.num_samples

        U = torch.zeros(num_samples, nbit, dtype=torch.float)
        B = torch.randn(num_train, nbit, dtype=torch.float)

        logging.info('Initializing training labels')
        Y = self._get_labels(self.dataloader['train_no_shuffle'])

        self.B = B.to(self.device)
        self.Y = Y.to(self.device)
        self.U = U.to(self.device)

        logging.info(f'self.U: {self.U.size()}; self.B: {self.B.size()}; self.Y: {self.Y.size()}')

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data

        if self.inference_datakey == 'db' and len(self.dataset['db']) == len(self.B):
            codes = self.B[index]
        else:
            image, labels = image.to(device), labels.to(device)
            with torch.no_grad():
                codes = self.model(image)

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

        train_codes = self.model(image)
        map_index = torch.tensor([self.randidxs[i] for i in index.tolist()], device=self.device)
        # loss = self.criterion(train_codes, self.B, self.S[map_index, :], index)
        # self.U[map_index, :] = train_codes.detach()

        loss = self.criterion(train_codes, self.B, self.S[index, :], map_index)
        self.U[index, :] = train_codes.detach()

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))

    def train_one_epoch(self, **kwargs):
        self.model.train()
        self.criterion.train()

        bs = self.config.batch_size
        inner_epochs = self.config.method_params.inner_epochs

        # Sample training data for cnn learning
        num_samples = self.config.method_params.num_samples
        # randsampler, randidxs = engine.get_random_sampler(num_samples, len(self.dataset['train']))
        # train_loader = engine.dataloader(self.dataset['train'], bs, drop_last=True, sampler=randsampler)

        randidxs = torch.randperm(len(self.dataset['train']))[:num_samples]
        subset_ds = subset_dataset(self.dataset['train'], randidxs)
        train_loader = engine.dataloader(subset_ds, bs, shuffle=True, drop_last=True)

        retrieval_labels = self.Y
        # if len(self.dataset['db']) == len(self.dataset['train']):
        train_labels = self.Y[randidxs]
        # else:
        #     subsetsampler = engine.get_sequential_sampler(randidxs)
        #     label_loader = engine.dataloader(self.dataset['train'], bs, drop_last=True, sampler=subsetsampler)
        #     train_labels = self._get_labels(label_loader).to(self.device)

        logging.info('Creating similarity matrix')
        # Create Similarity matrix
        S = (train_labels.unsqueeze(1) == retrieval_labels.unsqueeze(0)).float()  # num samples * train num
        # S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
        S = S.mul_(2).sub_(1)  # torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r
        self.S = S
        # self.randidxs = {v: i for i, v in enumerate(randidxs.tolist())}  # map to i
        self.randidxs = {i: v for i, v in enumerate(randidxs.tolist())}  # map to i

        meters = defaultdict(AverageMeter)

        with tqdm(range(inner_epochs), desc='Inner:', bar_format='{l_bar}{bar:10}{r_bar}') as iepochs:
            imeters = defaultdict(AverageMeter)
            for iepoch in iepochs:
                with tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}', leave=False) as tepoch:
                    for i, data in enumerate(tepoch):
                        self.train_one_batch(data, imeters, bidx=i, **kwargs)
                        tepoch.set_postfix({k: v.avg for k, v in imeters.items()})
                for k, v in imeters.items():
                    meters[k].update(v.avg)
                iepochs.set_postfix({k: v.avg for k, v in meters.items()})

        self.scheduler.step()

        logging.info('DCC')
        expand_U = torch.zeros(self.B.shape).to(self.device)
        expand_U[randidxs, :] = self.U
        self.B = solve_dcc(self.B, self.U, expand_U, S, self.B.size(1), self.config.method_params.gamma)

        return meters
