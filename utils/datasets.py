import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import transforms

ROOTDIR = os.environ.get('ROOTDIR', '.')  # for condor


def use_torch_loader():
    return lambda x: torch.load(x, map_location='cpu')


def use_pil_loader():
    return pil_loader


class HashingDataset(Dataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train.txt',
                 separate_multiclass=False,
                 path_prefix='',
                 loader=pil_loader,
                 selected_classes=[],
                 train_data=None,
                 train_labels=None,
                 verbose=True):
        self.loader = loader
        self.separate_multiclass = separate_multiclass
        self.root = os.path.expanduser(root)
        if isinstance(transform, (list, ListConfig)):
            transform = transforms.Compose(transform)
        if verbose:
            print(transform)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        if path_prefix != '' and path_prefix[-1] != '/':
            path_prefix = path_prefix + '/'
        self.path_prefix = path_prefix
        self.selected_classes = selected_classes
        self.verbose = verbose

        if train_data is None:
            self.train_data = []
            self.train_labels = []

            filename = os.path.join(self.root, self.filename)

            with open(filename, 'r') as f:
                while True:
                    lines = f.readline()
                    if not lines:
                        break

                    lines = lines.strip()
                    split_lines = lines.split()
                    path_tmp = split_lines[0]
                    label_tmp = split_lines[1:]
                    self.is_onehot = len(label_tmp) != 1
                    if not self.is_onehot:
                        label_tmp = label_tmp[0]
                    if self.separate_multiclass:
                        assert self.is_onehot, 'if multiclass, please use onehot'
                        nonzero_index = np.nonzero(np.array(label_tmp, dtype=np.int))[0]
                        for c in nonzero_index:
                            self.train_data.append(path_tmp)
                            label_tmp = ['1' if i == c else '0' for i in range(len(label_tmp))]
                            self.train_labels.append(label_tmp)
                    else:
                        self.train_data.append(path_tmp)
                        self.train_labels.append(label_tmp)

            self.train_data = np.array(self.train_data)
            self.train_labels = np.array(self.train_labels, dtype=np.float32)

            if len(selected_classes) != 0:
                cmask = np.zeros(self.train_data.shape[0], dtype=np.bool)
                if self.is_onehot:  # this do not work for multiclass
                    label_idx = self.train_labels.argmax(1).astype(np.int32)
                else:
                    label_idx = self.train_labels.astype(np.int32)
                for c in selected_classes:
                    cmask |= label_idx == c

                dump_number = self.train_data.shape[0] - cmask.sum()
                print(f'Dumped number of data: {dump_number}')
                self.train_data = self.train_data[cmask]
                self.train_labels = self.train_labels[cmask]

            print(f'Number of data: {self.train_data.shape[0]}')
        else:
            self.train_data = train_data
            self.train_labels = train_labels
            self.is_onehot = len(self.train_labels.shape) == 2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(os.path.join(ROOTDIR, f'{self.path_prefix}{path}'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.train_data)


class DescriptorDataset(Dataset):
    def __init__(self, root, filename, ratio=1, selected_classes=[]):
        self.data_dict = torch.load(os.path.join(root, filename), map_location=torch.device('cpu'))
        self.filename = filename
        self.root = root
        self.ratio = ratio

        if ratio != 1:
            assert 0 < ratio < 1, 'data ratio is in between 0 and 1 exclusively'
            N = len(self.data_dict['codes'])
            randidx = np.arange(N)
            np.random.shuffle(randidx)
            randidx = randidx[:int(ratio * N)]
            for key in self.data_dict:
                self.data_dict[key] = self.data_dict[key][randidx]

        if len(selected_classes) != 0:
            # assert len(self.data_dict['labels'].shape) == 1, 'not support for multi class'
            new_codes = []
            new_labels = []
            for c in selected_classes:
                if len(self.data_dict['labels'].size()) == 2:
                    cmask = self.data_dict['labels'].argmax(1) == c
                else:
                    cmask = self.data_dict['labels'] == c
                new_codes.append(self.data_dict['codes'][cmask])
                new_labels.append(self.data_dict['labels'][cmask])
            self.data_dict['codes'] = torch.cat(new_codes, dim=0)
            self.data_dict['labels'] = torch.cat(new_labels, dim=0)

        logging.info(f'Number of data in {filename}: {self.__len__()}')

    def __getitem__(self, index):
        embed = self.data_dict['codes'][index]
        label = self.data_dict['labels'][index]  # label is 1 indexed, convert to 0-indexed

        return embed, label, index  # img, None, index is throw error

    def __len__(self):
        return len(self.data_dict['codes'])


class IndexWrapperDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return tuple(list(self.dataset[index]) + [index])

    def __len__(self):
        return len(self.dataset)


class LandmarkDescriptorDataset(DescriptorDataset):

    def __getitem__(self, index):
        embed = self.data_dict['codes'][index]
        if self.filename == 'train.txt':
            label = self.data_dict['labels'][index] - 1  # label is 1 indexed, convert to 0-indexed
        else:
            label = 0
        landmark_id = self.data_dict['id'][index]

        return embed, label, (landmark_id, index)  # img, None, index is throw error

    def __len__(self):
        return len(self.data_dict['id'])


class GLDv2Dataset(HashingDataset):
    def __getitem__(self, index):
        img, label, _ = super(GLDv2Dataset, self).__getitem__(index)
        landmark_id = self.train_data[index].split('.jpg')[0].split('/')[-1]  # a/b/c/abcxxxx.jpg -> abcxxxx

        return img, label, (landmark_id, index)


class OneHot:
    def __init__(self, nclass):
        self.nclass = nclass

    def __call__(self, index):
        index = torch.tensor(int(index)).long()
        return F.one_hot(index, self.nclass)


def to_long(x):
    return x.long()


def one_hot(nclass):
    def f(index):
        index = torch.tensor(int(index)).long()
        return torch.nn.functional.one_hot(index, nclass)

    return f


def multi_label_to_one_hot(nclass):
    def f(index):
        one_hot = torch.zeros(nclass)
        one_hot[index] = 1
        return one_hot

    return f


def mnist(**kwargs):
    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    root = kwargs['root']

    if isinstance(transform, (list, ListConfig)):
        transform = transforms.Compose(transform)

    traind = MNIST(f'{root}',
                   transform=transform, target_transform=one_hot(10),
                   train=True, download=True)
    testd = MNIST(f'{root}', train=False, download=True)
    combine_data = torch.cat([traind.data, testd.data], dim=0).numpy()
    combine_targets = torch.cat([traind.targets, testd.targets], dim=0).numpy()
    is_train = torch.cat([torch.ones(len(traind.data)), torch.zeros(len(testd.targets))], dim=0).bool().numpy()

    path = f'{root}/{fn}'

    load_data = not os.path.exists(path)

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(combine_data.shape[0])  # [0, 1, ...]

        for i in range(10):
            class_mask = combine_targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            if ep == 1:
                query_n = 100  # // (nclass // 10)
                train_n = 500  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()

                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db[:train_n]
            elif ep == 2:
                query_n = 1000  # // (nclass // 10)
                train_n = 500  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()

                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db[:train_n]
            elif ep == 3:
                query_n = 1000  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()
                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db
            else:  # no shuffle
                train_n = 500

                index_for_query = data_id[(class_mask & (~is_train))].tolist()  # 1000
                index_for_db = data_id[(class_mask & is_train)]

                train_randidx = torch.randperm(len(index_for_db))[:train_n].numpy()
                index_for_train = index_for_db[train_randidx].tolist()
                index_for_db = index_for_db.tolist()

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        torch.save(train_data_index, f'{root}/{ep}_train.txt')
        torch.save(query_data_index, f'{root}/{ep}_test.txt')
        torch.save(db_data_index, f'{root}/{ep}_database.txt')
        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    traind.data = torch.from_numpy(combine_data[data_index])
    traind.targets = torch.from_numpy(combine_targets[data_index])

    return IndexWrapperDataset(traind)


def cifar(nclass, **kwargs):
    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    reset = kwargs['reset']

    root_prefix = kwargs['root']

    if isinstance(transform, (list, ListConfig)):
        transform = transforms.Compose(transform)

    print(transform)

    CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
    traind = CIFAR(f'{root_prefix}{nclass}',
                   transform=transform, target_transform=one_hot(int(nclass)),
                   train=True, download=True)
    testd = CIFAR(f'{root_prefix}{nclass}', train=False, download=True)

    combine_data = np.concatenate([traind.data, testd.data], axis=0)
    combine_targets = np.concatenate([traind.targets, testd.targets], axis=0)
    is_train = np.concatenate([np.ones(len(traind.data)), np.zeros(len(testd.targets))], axis=0).astype(bool)

    path = f'{root_prefix}{nclass}/0_{ep}_{fn}'

    load_data = (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(combine_data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = combine_targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            if ep == 1:
                query_n = 100  # // (nclass // 10)
                train_n = 500  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()

                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db[:train_n]
            elif ep == 2:
                query_n = 1000  # // (nclass // 10)
                train_n = 500  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()

                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db[:train_n]
            elif ep == 3:
                query_n = 1000  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()
                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db
            else:  # no shuffle
                train_n = 500

                index_for_query = data_id[(class_mask & (~is_train))].tolist()  # 1000
                index_for_db = data_id[(class_mask & is_train)]

                train_randidx = torch.randperm(len(index_for_db))[:train_n].numpy()
                index_for_train = index_for_db[train_randidx].tolist()
                index_for_db = index_for_db.tolist()

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        torch.save(train_data_index, f'{root_prefix}{nclass}/0_{ep}_train.txt')
        torch.save(query_data_index, f'{root_prefix}{nclass}/0_{ep}_test.txt')
        torch.save(db_data_index, f'{root_prefix}{nclass}/0_{ep}_database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    traind.data = combine_data[data_index]
    traind.targets = combine_targets[data_index]

    return IndexWrapperDataset(traind)


def read_class_names(path):
    names = open(path).readlines()
    names = [name.strip() for name in names]
    return names


def subset_dataset(dataset: HashingDataset, indices):
    train_data = dataset.train_data[indices]
    train_labels = dataset.train_labels[indices]
    new_dataset = HashingDataset(dataset.root,
                                 dataset.transform,
                                 dataset.target_transform,
                                 dataset.filename,
                                 dataset.separate_multiclass,
                                 dataset.path_prefix,
                                 dataset.loader,
                                 dataset.selected_classes,
                                 train_data,
                                 train_labels,
                                 verbose=False)
    return new_dataset
