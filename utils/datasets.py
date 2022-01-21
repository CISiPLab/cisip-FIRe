import logging
import os
from abc import ABC
from typing import Tuple, Any

import numpy as np
import torch
import torchvision
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader, accimage_loader
from torchvision.transforms import transforms
from tqdm import tqdm

import configs
from functions.evaluate_roxf import configdataset, DATASETS
from functions.mining import SimpleMemoryBank
from utils.augmentations import GaussianBlurOpenCV


class BaseDataset(Dataset, ABC):
    def get_img_paths(self):
        raise NotImplementedError


class HashingDataset(BaseDataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train',
                 separate_multiclass=False,
                 ratio=1):
        if torchvision.get_image_backend() == 'PIL':
            self.loader = pil_loader
        else:
            self.loader = accimage_loader

        self.separate_multiclass = separate_multiclass
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_data = []
        self.train_labels = []
        self.ratio = ratio

        filename = os.path.join(self.root, self.filename)

        is_pkl = False

        with open(filename, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break

                path_tmp = lines.split()[0]
                label_tmp = lines.split()[1:]
                self.is_onehot = len(label_tmp) != 1
                if not self.is_onehot:
                    label_tmp = lines.split()[1]
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

                is_pkl = path_tmp.endswith('.pkl')  # if save as pkl, pls make sure dont use different style of loading

        if is_pkl:
            self.loader = torch.load

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=float)

        if ratio != 1:
            assert 0 < ratio < 1, 'data ratio is in between 0 and 1 exclusively'
            N = len(self.train_data)
            randidx = np.arange(N)
            np.random.shuffle(randidx)
            randidx = randidx[:int(ratio * N)]
            self.train_data = self.train_data[randidx]
            self.train_labels = self.train_labels[randidx]

        logging.info(f'Number of data: {self.train_data.shape[0]}')

    def filter_classes(self, classes):  # only work for single class dataset
        new_data = []
        new_labels = []

        for idx, c in enumerate(classes):
            new_onehot = np.zeros(len(classes))
            new_onehot[idx] = 1
            cmask = self.train_labels.argmax(axis=1) == c

            new_data.append(self.train_data[cmask])
            new_labels.append(np.repeat([new_onehot], int(np.sum(cmask)), axis=0))
            # new_labels.append(self.train_labels[cmask])

        self.train_data = np.concatenate(new_data)
        self.train_labels = np.concatenate(new_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)

    def get_img_paths(self):
        return self.train_data


class IndexDatasetWrapper(BaseDataset):
    def __init__(self, ds) -> None:
        super(Dataset, self).__init__()
        self.__dict__['ds'] = ds

    def __setattr__(self, name, value):
        setattr(self.ds, name, value)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ds, attr)

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        outs = self.ds.__getitem__(index)
        return tuple(list(outs) + [index])

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class InstanceDiscriminationDataset(BaseDataset):
    def augment_image(self, img):
        # if use this, please run script with --no-aug and --gpu-mean-transform
        return self.transform(self.to_pil(img))

    def weak_augment_image(self, img):
        # if use this, please run script with --no-aug and --gpu-mean-transform
        return self.weak_transform(self.to_pil(img))

    def __init__(self, ds, tmode='simclr', imgsize=224, weak_mode=0) -> None:
        super(Dataset, self).__init__()
        self.__dict__['ds'] = ds

        if 'simclr' in tmode:
            s = 0.5
            size = imgsize
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.7),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlurOpenCV(kernel_size=3),
                                                  # GaussianBlur(kernel_size=int(0.1 * size)),
                                                  transforms.ToTensor(),
                                                  # 0.2 * 224 = 44 pixels
                                                  transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))])
            self.transform = data_transforms

        # lazy fix, can be more pretty and general, cibhash part 1/2
        elif tmode == 'cibhash':
            logging.info('CIBHash Augmentations')
            s = 0.5
            size = imgsize
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.7),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlurOpenCV(kernel_size=3),
                                                  # GaussianBlur(kernel_size=3),
                                                  transforms.ToTensor()])
            self.transform = data_transforms

        else:
            raise ValueError(f'unknown mode {tmode}')

        if weak_mode == 1:
            logging.info(f'Weak mode {weak_mode} activated.')
            self.weak_transform = transforms.Compose([
                transforms.Resize(256),  # temp lazy hard code
                transforms.CenterCrop(imgsize),
                transforms.ToTensor()
            ])
        elif weak_mode == 2:
            logging.info(f'Weak mode {weak_mode} activated.')
            self.weak_transform = transforms.Compose([
                transforms.Resize(256),  # temp lazy hard code
                transforms.RandomCrop(imgsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        self.weak_mode = weak_mode
        self.tmode = tmode
        self.imgsize = imgsize
        self.to_pil = transforms.ToPILImage()

    def __setattr__(self, name, value):
        setattr(self.ds, name, value)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ds, attr)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        out = self.ds.__getitem__(index)
        img, target = out[:2]  # exclude index

        # if self.tmode == 'simclr':
        #     aug_imgs = [img, self.augment_image(img)]
        # else:
        if self.weak_mode != 0:
            aug_imgs = [self.weak_augment_image(img), self.augment_image(img)]
        else:
            aug_imgs = [self.augment_image(img), self.augment_image(img)]

        return torch.stack(aug_imgs, dim=0), target, index

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


class RotationDataset(BaseDataset):

    @staticmethod
    def rotate_img(img, rot):
        img = np.transpose(img.numpy(), (1, 2, 0))
        if rot == 0:  # 0 degrees rotation
            out = img
        elif rot == 90:  # 90 degrees rotation
            out = np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            out = np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            out = np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')
        return torch.from_numpy(np.transpose(out, (2, 0, 1)).copy())

    def __init__(self, ds) -> None:
        super(Dataset, self).__init__()
        self.__dict__['ds'] = ds

    def __setattr__(self, name, value):
        setattr(self.ds, name, value)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ds, attr)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        out = self.ds.__getitem__(index)
        img, target = out[:2]  # exclude index

        # rot_label = np.random.randint(0, 4)  # .item()
        rot_labels = [0, 1, 2, 3]

        rots = [0, 90, 180, 270]
        # rots = [0, rots[rot_label]]
        rot_imgs = [self.rotate_img(img, rot) for rot in rots]

        return torch.stack(rot_imgs, dim=0), torch.tensor(rot_labels), target, index

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


class LandmarkDataset(BaseDataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train.csv',
                 onehot=False, return_id=False):
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_labels = []
        self.set_name = filename[:-4]
        self.onehot = onehot
        self.return_id = return_id

        def get_path(i: str):
            return os.path.join(root, self.set_name, i[0], i[1], i[2], i + ".jpg")

        filename = os.path.join(self.root, self.filename)
        self.df = read_csv(filename)
        self.df['path'] = self.df['id'].apply(get_path)
        self.max_index = self.df['landmark_id'].max() + 1

        logging.info(f'Number of data: {len(self.df)}')

    def to_onehot(self, i):
        t = torch.zeros(self.max_index)
        t[i] = 1
        return t

    def __getitem__(self, index):
        img = self.df['path'][index]

        if self.onehot:
            target = self.to_onehot(self.df['landmark_id'][index])
        else:
            target = self.df['landmark_id'][index]
        # target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_id:
            return img, target, (self.df['id'][index], index)
        return img, target

    def __len__(self):
        return len(self.df)

    def get_img_paths(self):
        return self.df['path'].to_numpy()


class SingleIDDataset(BaseDataset):
    """Dataset with only single class ID
    To be merge with Landmark"""

    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train.csv',
                 onehot=False):
        self.loader = pil_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_labels = []
        self.set_name = filename[:-4]
        self.onehot = onehot

        def get_path(i: str):
            return os.path.join(root, "imgs", i)

        filename = os.path.join(self.root, self.filename)
        self.df = read_csv(filename)
        self.df['path'] = self.df['path'].apply(get_path)
        self.max_index = self.df['class_id'].max() + 1

        logging.info(f'Number of data: {len(self.df)}')

    def to_onehot(self, i):
        t = torch.zeros(self.max_index)
        t[i] = 1
        return t

    def __getitem__(self, index):
        img = self.df['path'][index]

        if self.onehot:
            target = self.to_onehot(self.df['class_id'][index])
        else:
            target = self.df['class_id'][index]
        # target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.df)

    def get_img_paths(self):
        return self.df['path'].to_numpy()


class ROxfordParisDataset(BaseDataset):
    def __init__(self,
                 dataset='roxford5k',
                 filename='test.txt',
                 transform=None,
                 target_transform=None):
        self.loader = pil_loader
        self.transform = transform
        self.target_transform = target_transform
        assert filename in ['test.txt', 'database.txt']
        self.set_name = filename
        assert dataset in DATASETS
        self.cfg = configdataset(dataset, os.path.join('data'))

        logging.info(f'Number of data: {self.__len__()}')

    def __getitem__(self, index):
        if self.set_name == 'database.txt':
            img = self.cfg['im_fname'](self.cfg, index)
        elif self.set_name == 'test.txt':
            img = self.cfg['qim_fname'](self.cfg, index)

        img = self.loader(img)
        if self.set_name == 'test.txt':
            img = img.crop(self.cfg['gnd'][index]['bbx'])

        if self.transform is not None:
            img = self.transform(img)

        return img, index, index  # img, None, index is throw error

    def __len__(self):
        if self.set_name == 'test.txt':
            return self.cfg['nq']
        elif self.set_name == 'database.txt':
            return self.cfg['n']

    def get_img_paths(self):
        raise NotImplementedError('Not supported.')


class DescriptorDataset(BaseDataset):
    def __init__(self, root, filename, ratio=1):
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

        logging.info(f'Number of data in {filename}: {self.__len__()}')

    def __getitem__(self, index):
        embed = self.data_dict['codes'][index]
        label = self.data_dict['labels'][index]  # label is 1 indexed, convert to 0-indexed

        return embed, label, index  # img, None, index is throw error

    def __len__(self):
        return len(self.data_dict['codes'])

    def get_img_paths(self):
        raise NotImplementedError('Not supported for descriptor dataset. Please try usual Image Dataset if you want to get all image paths.')


class EmbeddingDataset(BaseDataset):
    def __init__(self, root,
                 filename='train.txt'):
        self.data_dict = torch.load(os.path.join(root, filename), map_location=torch.device('cpu'))
        self.filename = filename
        self.root = root
        logging.info(f'Number of data in {filename}: {self.__len__()}')

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

    def get_img_paths(self):
        raise NotImplementedError('Not supported for descriptor dataset. Please try usual Image Dataset if you want to get all image paths.')


class NeighbourDatasetWrapper(BaseDataset):
    def __init__(self, ds, model, config) -> None:
        super(Dataset, self).__init__()
        self.ds = ds

        device = config['device']
        loader = DataLoader(ds, config['batch_size'],
                            shuffle=False,
                            drop_last=False,
                            num_workers=os.cpu_count())

        model.eval()
        pbar = tqdm(loader, desc='Obtain Codes', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                    disable=configs.disable_tqdm)
        ret_feats = []

        for i, (data, labels, index) in enumerate(pbar):
            with torch.no_grad():
                data, labels = data.to(device), labels.to(device)
                x, code_logits, b = model(data)[:3]
                ret_feats.append(x.cpu())

        ret_feats = torch.cat(ret_feats)

        mbank = SimpleMemoryBank(len(self.ds), model.backbone.in_features, device)
        mbank.update(ret_feats)

        neighbour_topk = config['dataset_kwargs'].get('neighbour_topk', 5)
        indices = mbank.mine_nearest_neighbors(neighbour_topk)

        self.indices = indices[:, 1:]  # exclude itself

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.ds.__getitem__(index)

        randidx = np.random.choice(self.indices[index], 1)[0]
        nbimg, nbtar = self.ds.__getitem__(randidx)

        return img, target, index, nbimg, nbtar, randidx

    def __len__(self):
        return len(self.ds)

    def get_img_paths(self):
        return self.ds.get_img_paths()


def one_hot(nclass):
    def f(index):
        index = torch.tensor(int(index)).long()
        return torch.nn.functional.one_hot(index, nclass)

    return f


def cifar(nclass, **kwargs):
    transform = kwargs['transform']
    ep = kwargs['evaluation_protocol']
    fn = kwargs['filename']
    reset = kwargs['reset']

    CIFAR = CIFAR10 if int(nclass) == 10 else CIFAR100
    traind = CIFAR(f'data/cifar{nclass}',
                   transform=transform, target_transform=one_hot(int(nclass)),
                   train=True, download=True)
    traind = IndexDatasetWrapper(traind)
    testd = CIFAR(f'data/cifar{nclass}',
                  transform=transform, target_transform=one_hot(int(nclass)),
                  train=False, download=True)
    testd = IndexDatasetWrapper(testd)

    if ep == 2:  # using orig train and test
        if fn == 'test.txt':
            return testd
        else:  # train.txt and database.txt
            return traind

    combine_data = np.concatenate([traind.data, testd.data], axis=0)
    combine_targets = np.concatenate([traind.targets, testd.targets], axis=0)

    path = f'data/cifar{nclass}/0_0_{ep}_{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        logging.info(f'Loading {path}')
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
            elif ep == 2:  # ep2 = take all data
                query_n = 1000  # // (nclass // 10)

                index_for_query = index_of_class[:query_n].tolist()
                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db

            elif ep == 3:  # Bi-Half Cifar10(II)
                query_n = 1000
                train_n = 500
                index_for_query = index_of_class[:query_n].tolist()
                index_for_db = index_of_class[query_n:].tolist()
                index_for_train = index_for_db[:train_n]

            else:
                raise NotImplementedError('')

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)

        torch.save(train_data_index, f'data/cifar{nclass}/0_0_{ep}_train.txt')
        torch.save(query_data_index, f'data/cifar{nclass}/0_0_{ep}_test.txt')
        torch.save(db_data_index, f'data/cifar{nclass}/0_0_{ep}_database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    traind.data = combine_data[data_index]
    traind.targets = combine_targets[data_index]

    return traind


def imagenet100(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/imagenet{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def cars(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = HashingDataset('data/cars', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def landmark(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    return_id = kwargs.get('return_id', False)

    d = LandmarkDataset('data/landmark', transform=transform, filename=filename, return_id=return_id)
    return d


def nuswide(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    separate_multiclass = kwargs.get('separate_multiclass', False)
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/nuswide_v2_256{suffix}',
                       transform=transform,
                       filename=filename,
                       separate_multiclass=separate_multiclass,
                       ratio=kwargs.get('ratio', 1))
    return d


def nuswide_single(**kwargs):
    return nuswide(separate_multiclass=True, **kwargs)


def coco(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/coco{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def roxford5k(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    d = ROxfordParisDataset(dataset='roxford5k', filename=filename, transform=transform)
    return d


def rparis6k(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    d = ROxfordParisDataset(dataset='rparis6k', filename=filename, transform=transform)
    return d


def gldv2delgembed(**kwargs):
    filename = kwargs['filename']
    d = EmbeddingDataset('data/gldv2delgembed', filename=filename)
    return d


def roxford5kdelgembed(**kwargs):
    filename = kwargs['filename']
    d = EmbeddingDataset('data/roxford5kdelgembed', filename=filename)
    return d


def rparis6kdelgembed(**kwargs):
    filename = kwargs['filename']
    d = EmbeddingDataset('data/rparis6kdelgembed', filename=filename)
    return d


def descriptor(**kwargs):
    filename = kwargs['filename']
    data_folder = kwargs['data_folder']
    d = DescriptorDataset(data_folder, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def mirflickr(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/mirflickr{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def sop_instance(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = SingleIDDataset('data/sop_instance', transform=transform, filename=filename)
    return d


def sop(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']
    suffix = kwargs.get('dataset_name_suffix', '')

    d = HashingDataset(f'data/sop{suffix}', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d


def food101(**kwargs):
    transform = kwargs['transform']
    filename = kwargs['filename']

    d = HashingDataset('data/food-101', transform=transform, filename=filename, ratio=kwargs.get('ratio', 1))
    return d
