import logging
import os
import random

import numpy as np
import torch
import torchvision
from torch.optim import lr_scheduler
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import models
from functions.optim.adan import Adan
from utils import datasets
from utils.augmentations import get_train_transform

if torch.cuda.device_count() != 0:
    default_workers = os.cpu_count() // torch.cuda.device_count()  # follow PyTorch recommendation
else:
    default_workers = os.cpu_count()

# huge in term of number of classes
non_onehot_dataset = ['landmark', 'gldv2delgembed', 'sop_instance',
                      'sop_instance_alexnet', 'sop_instance_vgg16', 'sop_instance_resnet18']
dataset_evaluated_by_id = ['landmark', 'gldv2delgembed']
embedding_datasets = ['gldv2delgembed', 'roxford5kdelgembed' 'rparis6kdelgembed']

pin_memory = False
disable_tqdm = False


def in_features(dfolder, dataset):
    if dataset == 'gldv2delgembed':
        return 2048
    elif dataset == 'descriptor':
        if dfolder == '':
            return 0
        if '128' in dfolder:
            return 128
        elif 'alexnet' in dfolder or 'vgg' in dfolder:
            return 4096
        else:
            return 512


def imagesize(config):
    if not isinstance(config, dict):
        dsname = config
    else:
        dsname = config['dataset']

    r = {
        'imagenet100': 256,
        'nuswide': 256,
        'coco': 256,
        'cifar10': 224,
        'cifar10_2': 224,
        'cifar10_II': 224,
        'cars': 224,
        'landmark': 512,
        'roxford5k': 224,
        'rparis6k': 224,
        'gldv2delgembed': 0,
        'roxford5kdelgembed': 0,
        'mirflickr': 256,
        'sop': 256,
        'sop_instance': 256,
        'food101': 256
    }[dsname]

    return r


def cropsize(config):
    if not isinstance(config, dict):
        dsname = config
    else:
        dsname = config['dataset']

    r = {
        'imagenet100': 224,
        'nuswide': 224,
        'coco': 224,
        'cifar10': 224,
        'cifar10_2': 224,
        'cifar10_II': 224,
        'cars': 224,
        'landmark': 512,
        'roxford5k': 224,
        'rparis6k': 224,
        'gldv2delgembed': 0,
        'roxford5kdelgembed': 0,
        'rparis6kdelgembed': 0,
        'mirflickr': 224,
        'sop': 224,
        'sop_instance': 224,
        'food101': 224
    }[dsname]

    return r


def nclass(config):
    if not isinstance(config, dict):
        dsname = config
    else:
        dsname = config['dataset']

    r = {
        'imagenet100': 100,
        'cifar10': 10,
        'cifar10_2': 10,
        'cifar10_II': 10,
        'nuswide': 21,
        'coco': 80,
        'cars': 196,
        'landmark': 81313,
        'gldv2delgembed': 81313,  # same as landmark
        'roxford5kdelgembed': 0,  # not applicable
        'rparis6kdelgembed': 0,
        'mirflickr': 24,
        'sop': 12,
        'sop_instance': 22634,
        'food101': 101
    }[dsname]

    return r


def R(config):
    r = {
        'imagenet100': 1000,
        'cifar10': 59000,
        'cifar10_2': 50000,
        'cifar10_II': 50000,
        'nuswide': 5000,
        'coco': 5000,
        'cars': 100,
        'landmark': 100,
        'roxford5k': 0,  # not using
        'rparis6k': 0,  # not using
        'gldv2delgembed': 100,  # same as landmark
        'roxford5kdelgembed': 0,  # not using
        'rparis6kdelgembed': 0,
        'mirflickr': 1000,
        'sop': 1000,
        'sop_instance': 100,
        'food101': 1000
    }[config['dataset'] + {2: '_2'}.get(config['dataset_kwargs']['evaluation_protocol'], '')]

    return r


def arch(config, **kwargs):
    if config['arch'] in models.network_names:
        net = models.network_names[config['arch']](config, **kwargs)
    else:
        raise ValueError(f'Invalid Arch: {config["arch"]}')

    return net


def optimizer(config, params):
    o_type = config['optim']
    kwargs = config['optim_kwargs']

    if o_type == 'sgd':
        o = SGD(params,
                lr=kwargs['lr'],
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0005),
                nesterov=kwargs.get('nesterov', False))
    elif o_type == 'rmsprop':
        o = RMSprop(params,
                    lr=kwargs['lr'],
                    alpha=kwargs.get('alpha', 0.99),
                    weight_decay=kwargs.get('weight_decay', 0.0005),
                    momentum=kwargs.get('momentum', 0))
    elif o_type == 'adam':  # adam
        o = Adam(params,
                 lr=kwargs['lr'],
                 betas=kwargs.get('betas', (0.9, 0.999)),
                 weight_decay=kwargs.get('weight_decay', 0))
    elif o_type == 'adan':
        o = Adan(params,
                 lr=kwargs['lr'],
                 betas=[0.98, 0.92, 0.99], # optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)
                 weight_decay=0.02,  # weight decay, similar one used in AdamW (default: 0.02)
                 eps=1e-8, # optimizer epsilon to avoid the bad case where second-order moment is zero
                 ## (default: None, use opt default 1e-8 in adan)
                 max_grad_norm=0.0, # if the l2 norm is large than this hyper-parameter,
                 # then we clip the gradient  (default: 0.0, no gradient clip)
                 no_prox=False #whether perform weight decay like AdamW (default=False)
                 )
    else:
        raise ValueError(f'Optimizer specified {o_type} is not defined.')

    return o


def scheduler(config, optimizer):
    s_type = config['scheduler']
    kwargs = config['scheduler_kwargs']

    if s_type == 'step':
        return lr_scheduler.StepLR(optimizer,
                                   kwargs['step_size'],
                                   kwargs['gamma'])
    elif s_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer,
                                        [int(float(m) * int(config['epochs'])) for m in
                                         kwargs['milestones'].split(',')],
                                        kwargs['gamma'])
    elif s_type == 'linear':
        def function(e):
            init_lr = kwargs['linear_init_lr']
            last_lr = kwargs['linear_last_lr']
            epochs = config['epochs']
            return ((last_lr - init_lr) / (epochs - 1) * e + init_lr) / init_lr  # y = mx + c

        return lr_scheduler.LambdaLR(optimizer, function)
    else:
        raise Exception('Scheduler not supported yet: ' + s_type)


def get_meanstd(norm):
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }[norm]
    return mean, std


def compose_transform(mode='train', resize=0, crop=0, norm=0,
                      augmentations=None):
    """

    :param mode: train/test
    :param resize: if 0, will not add Resize
    :param crop: if 0, will not add CenterCrop (only for test mode)
    :param norm: if 2, will not add Normalize
    :param augmentations: augmentation Compose (only for train mode)
    :return: Compose list [(Resize), (train:augmentations), (test:CenterCrop), ToTensor, (Normalize)]
    # () based on options
    """
    # norm = 0, 0 to 1
    # norm = 1, -1 to 1
    # norm = 2, standardization
    mean, std = get_meanstd(norm)

    compose = []

    if resize != 0:
        compose.append(transforms.Resize((resize, resize)))

    if mode == 'train' and augmentations is not None:
        compose += augmentations

    if mode == 'test' and crop != 0 and resize != crop:
        compose.append(transforms.CenterCrop(crop))

    compose.append(transforms.ToTensor())

    if norm != 0:
        compose.append(transforms.Normalize(mean, std))

    return transforms.Compose(compose)


def dataset(config, filename, transform_mode,
            return_id=False, gpu_transform=False, gpu_mean_transform=False,
            skip_preprocess=False):
    dataset_name = config['dataset']
    use_db_as_train = config['dataset_kwargs'].get('use_db_as_train', False)

    data_ratio = 1
    if filename == 'train.txt' and use_db_as_train:
        filename = 'database.txt'
        data_ratio = config['dataset_kwargs'].get('train_ratio', 1)

    nclass = config['arch_kwargs']['nclass']

    resize = config['dataset_kwargs'].get('resize', 0)
    crop = config['dataset_kwargs'].get('crop', 0)
    norm = config['dataset_kwargs'].get('norm', 1)
    use_rand_aug = config['dataset_kwargs']['use_random_augmentation']
    reset = config['dataset_kwargs'].get('reset', False)
    remove_train_from_db = config['dataset_kwargs'].get('remove_train_from_db', False)
    separate_multiclass = config['dataset_kwargs'].get('separate_multiclass', False)
    extra_dataset = config['dataset_kwargs'].get('extra_dataset', 0)

    if dataset_name in ['imagenet100', 'nuswide', 'coco', 'cars', 'landmark',
                        'roxford5k', 'rparis6k', 'mirflickr', 'sop', 'sop_instance', 'food101']:
        norm = 2 if not gpu_mean_transform else 0  # 0 = turn off Normalize

        if skip_preprocess:  # will not resize and crop, and no augmentation
            transform = compose_transform('test', 0, 0, norm)
        else:
            if transform_mode == 'train':
                transform = compose_transform('train', 0, crop, norm,
                                              get_train_transform(dataset_name, resize, crop, use_rand_aug))
            else:
                transform = compose_transform('test', resize, crop, norm)

        datafunc = {
            'imagenet100': datasets.imagenet100,
            'nuswide': datasets.nuswide,
            'coco': datasets.coco,
            'cars': datasets.cars,
            'landmark': datasets.landmark,
            'roxford5k': datasets.roxford5k,
            'rparis6k': datasets.rparis6k,
            'mirflickr': datasets.mirflickr,
            'sop': datasets.sop,
            'sop_instance': datasets.sop_instance,
            'food101': datasets.food101
        }[dataset_name]
        d = datafunc(transform=transform,
                     filename=filename,
                     separate_multiclass=separate_multiclass,
                     return_id=return_id,
                     dataset_name_suffix=config['dataset_kwargs'].get('dataset_name_suffix', ''),
                     ratio=data_ratio)
        logging.info(f'Augmentation for {transform_mode}: {transform.transforms}')

    elif dataset_name in ['cifar10', 'cifar100', 'cifar10_II']:  # cifar10/ cifar100
        resizec = 0 if resize == 32 else resize
        cropc = 0 if crop == 32 else crop

        norm = 2 if not gpu_mean_transform else 0  # 0 = turn off Normalize

        if skip_preprocess:  # cifar10 will always resize first
            transform = compose_transform('test', resizec, 0, norm)
        else:
            if transform_mode == 'train':
                transform = compose_transform('train', resizec, 0, norm, [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.05, contrast=0.05),
                ])
            else:
                transform = compose_transform('test', resizec, cropc, norm)

        ep = config['dataset_kwargs'].get('evaluation_protocol', 1)
        if dataset_name == 'cifar10_II':
            ep = 3

        if dataset_name in ['cifar10', 'cifar10_II', 'cifar100']:
            d = datasets.cifar(nclass, transform=transform, filename=filename, evaluation_protocol=ep, reset=reset,
                               remove_train_from_db=remove_train_from_db, extra_dataset=extra_dataset)
            logging.info(f'Number of data: {len(d.data)}')
            logging.info(f'Augmentation for {transform_mode}: {transform.transforms}')
        else:
            raise NotImplementedError(f"Not implementation for {dataset_name}")
    elif dataset_name in ['gldv2delgembed', 'roxford5kdelgembed', 'rparis6kdelgembed']:
        datafunc = {
            'gldv2delgembed': datasets.gldv2delgembed,
            'roxford5kdelgembed': datasets.roxford5kdelgembed,
            'rparis6kdelgembed': datasets.rparis6kdelgembed
        }[dataset_name]
        d = datafunc(filename=filename)
    elif dataset_name == 'descriptor':  # descriptor
        data_folder = config['dataset_kwargs']['data_folder']
        d = datasets.descriptor(data_folder=data_folder,
                                filename=filename,
                                ratio=data_ratio)
    else:
        raise NotImplementedError(f"No implementation for {dataset_name}")

    return d


def dataloader(d, bs=256, shuffle=True, workers=-1, drop_last=True, collate_fn=None, seed=-1):
    """

    :param d:
    :param bs:
    :param shuffle:
    :param workers:
    :param drop_last:
    :param collate_fn:
    :param seed: random seed for deterministic
    :return:
    """
    if workers < 0:
        workers = default_workers
    if seed != -1:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    l = DataLoader(d,
                   bs,
                   shuffle,
                   drop_last=drop_last,
                   num_workers=workers,
                   pin_memory=pin_memory,
                   collate_fn=collate_fn,
                   worker_init_fn=seed_worker,
                   generator=g)
    return l


def seeding(seed):
    seed = int(seed)
    if seed != -1:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
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


def use_accimage_backend():
    torchvision.set_image_backend('accimage')
