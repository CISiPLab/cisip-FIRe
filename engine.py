import os
import random
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler

default_workers = os.cpu_count()


class SubsetSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for i in torch.arange(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


def get_random_sampler(num_samples, total_size):
    random_idxs = torch.randperm(total_size)[:num_samples]
    return SubsetRandomSampler(random_idxs), random_idxs


def get_sequential_sampler(idxs):
    return SubsetSampler(idxs)


def dataloader(d, bs=256, shuffle=False, workers=-1, drop_last=False, sampler=None):
    if len(d) == 0:  # if empty dataset, return empty list
        return []

    if workers < 0:
        workers = default_workers
    l = DataLoader(d,
                   bs,
                   shuffle,
                   drop_last=drop_last,
                   num_workers=workers,
                   sampler=sampler)
    return l


def seeding(seed):
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def tensor_to_dataset(tensor, transform=None):
    class TransformTensorDataset(Dataset):
        def __init__(self, tensor, ts=None):
            super(TransformTensorDataset, self).__init__()
            self.tensor = tensor
            self.ts = ts

        def __getitem__(self, index):
            if self.ts is not None:
                return self.ts(self.tensor[index])
            return self.tensor[index]

        def __len__(self):
            return len(self.tensor)

    ttd = TransformTensorDataset(tensor, transform)
    return ttd


def tensors_to_dataset(tensors_with_transform):
    """

    :param tensors_with_transform:
    [
        {
            'tensor': torch.Tensor,   # required
            'transform': callable,    # optional
        }, ...
    ]
    :return:
    """

    class TransformTensorDataset(Dataset):
        def __init__(self, tensors_with_ts):
            super(TransformTensorDataset, self).__init__()

            self.tensors_with_ts = tensors_with_ts

        def __getitem__(self, index):
            rets = []
            for tensor_dict in self.tensors_with_ts:
                tensor = tensor_dict['tensor']
                ts = tensor_dict.get('transform')
                if ts is not None:
                    rets.append(ts(tensor[index]))
                else:
                    rets.append(tensor[index])
            return rets

        def __len__(self):
            return len(self.tensors_with_ts[0]['tensor'])

    ttd = TransformTensorDataset(tensors_with_transform)
    return ttd
