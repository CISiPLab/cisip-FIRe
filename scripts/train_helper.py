import logging

import torch
import torch.nn as nn

import configs
from functions.hashing import get_hadamard
from functions.loss.adsh import ADSHLoss
from functions.loss.bihalf import BiHalfLoss
from functions.loss.ce import CELoss
from functions.loss.cibhash import CIBHashLoss
from functions.loss.csq import CSQLoss
from functions.loss.dbdh import DBDHLoss
from functions.loss.dfh import DFHLoss
from functions.loss.dpn import DPNLoss, get_centroid
from functions.loss.dpsh import DPSHLoss
from functions.loss.dtsh import DTSHLoss
from functions.loss.greedyhash import GreedyHashLoss, GreedyHashUnsupervisedLoss
from functions.loss.hashnet import HashNetLoss
from functions.loss.imh import IMHLoss
from functions.loss.itq import ITQLoss
from functions.loss.jmlh import JMLH
from functions.loss.lsh import LSHLoss
from functions.loss.mihash import MIHashLoss
from functions.loss.orthoarc import OrthoArcLoss
from functions.loss.orthocos import OrthoCosLoss
from functions.loss.pca import PCALoss
from functions.loss.sdh import SDHLossC, SDHLoss
from functions.loss.sh import SHLoss
from functions.loss.ssdh import SemanticStructureDHLoss
from functions.loss.tbh import TBHLoss
from utils.datasets import RotationDataset, InstanceDiscriminationDataset
from utils.misc import DataParallelPassthrough


def get_loss(loss_name, **cfg):
    loss = {
        'greedyhash': GreedyHashLoss,
        'dpn': DPNLoss,
        'orthocos': OrthoCosLoss,
        'ce': CELoss,
        'bihalf-supervised': CELoss,
        'orthoarc': OrthoArcLoss,
        'jmlh': JMLH,
        'sdhc': SDHLossC,
        'csq': CSQLoss,
        'adsh': ADSHLoss,
        'greedyhash-unsupervised': GreedyHashUnsupervisedLoss,
        'bihalf': BiHalfLoss,
        'hashnet': HashNetLoss,
        'dpsh': DPSHLoss,
        'dbdh': DBDHLoss,
        'mihash': MIHashLoss,
        'sdh': SDHLoss,
        'dfh': DFHLoss,
        'dtsh': DTSHLoss,
        'ssdh': SemanticStructureDHLoss,
        'tbh': TBHLoss,
        'itq': ITQLoss,
        'pca': PCALoss,
        'lsh': LSHLoss,
        'sh': SHLoss,
        'cibhash': CIBHashLoss,
        'imh': IMHLoss
    }
    if loss_name not in loss:
        raise NotImplementedError(f'not implemented for {loss_name}')
    return loss[loss_name](**cfg)


def update_criterion(model, criterion, loss_name, method, onehot):
    if loss_name in ['dpn', 'csq']:
        criterion.centroids = model.centroids
    elif loss_name in ['sdhc', 'sdh']:
        if isinstance(model.hash_fc, nn.Sequential):
            criterion.weight = model.hash_fc[0].weight
        else:
            criterion.weight = model.hash_fc.weight
    elif loss_name in ['adsh']:
        criterion.weight = model.ce_fc.centroids

    # update criterion as non-onehot mode, for pairwise methods
    if method in ['pairwise']:

        if not onehot and not criterion.label_not_onehot:
            logging.info("Not a onehot label dataset")
            criterion.label_not_onehot = True


def generate_centroids(nclass, nbit, init_method):
    if init_method == 'N':
        centroids = torch.randn(nclass, nbit)
    elif init_method == 'U':
        centroids = torch.rand(nclass, nbit) - 0.5
    elif init_method == 'B':
        prob = torch.ones(nclass, nbit) * 0.5
        centroids = torch.bernoulli(prob) * 2. - 1.
    elif init_method == 'M':
        centroids = get_centroid(nclass, nbit)
    elif init_method == 'H':
        centroids = get_hadamard(nclass, nbit)
    else:  # qr decomposition
        raise NotImplementedError('qr decomposition not implemented yet')
    return centroids


def get_dataloader(config,
                   filename,
                   shuffle=False,
                   drop_last=False,
                   gpu_transform=None,
                   gpu_mean_transform=None,
                   no_augmentation=False,
                   skip_preprocess=False,
                   dataset_type='',
                   full_batchsize=False,
                   workers=-1,
                   seed=-1):
    ds = configs.dataset(config,
                         filename=filename,
                         transform_mode='test' if no_augmentation else 'train',
                         gpu_transform=gpu_transform,
                         gpu_mean_transform=gpu_mean_transform,
                         skip_preprocess=skip_preprocess)
    if dataset_type == 'rotation':
        logging.info('Using Rotation Dataset!')
        ds = RotationDataset(ds)
    elif dataset_type == 'simclr':
        logging.info('Using SimCLR Dataset!')
        ds = InstanceDiscriminationDataset(ds,
                                           'simclr',
                                           config['dataset_kwargs']['crop'],
                                           config['dataset_kwargs']['weak_aug'])
    # lazy fix, can be more pretty and general, cibhash part 2/2
    elif dataset_type == 'cibhash':
        logging.info('Using CIBHash Dataset!')
        ds = InstanceDiscriminationDataset(ds, 'cibhash', config['dataset_kwargs']['crop'])

    if full_batchsize:
        logging.info('Loading full training dataset! Please be aware of memory usage')
        batch_size = len(ds)
    else:
        batch_size = config['batch_size']

    loader = configs.dataloader(ds,
                                batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                workers=workers,
                                seed=seed)
    return loader


def prepare_dataloader(config,
                       train_shuffle=True,
                       test_shuffle=False,
                       gpu_transform=False,
                       gpu_mean_transform=False,
                       include_train=True,
                       train_drop_last=True,
                       workers=-1,
                       train_full=False,
                       seed=-1):
    """

    :param config:
    {
        'arch': '',
        'arch_kwargs': {
            'nclass': 100,
        }
        'batch_size': 64,
        'max_batch_size': 64,
        'dataset': 'imagenet100' ,  # 'descriptor'
        'dataset_kwargs': {
            'resize': 224,
            'crop': 224,
            'norm': 2,
            'evaluation_protocol': 1,
            'use_db_as_train': False,
            'train_ratio': 1,
            'reset': False,
            'separate_multiclass': False,
            'train_skip_preprocess': False,
            'db_skip_preprocess': False,
            'test_skip_preprocess': False,
            'dataset_name_suffix': '',  # e.g. "_resize", it will load "data/xxx_resize"
            'neighbour_topk': 5,  # for neighbour dataset
            'no_augmentation': False,
            'data_folder': '',
            'dataset_type': ''
            'weak_aug': 0
        },
    }
    :param train_shuffle:
    :param test_shuffle:
    :param gpu_transform:
    :param gpu_mean_transform:
    :param include_train:
    :param train_drop_last:
    :param workers:
    :param train_full:
    :param seed:
    :return:
    """
    logging.info('Creating Datasets')
    if include_train:
        no_augmentation = config['dataset_kwargs'].get('no_augmentation', False)
        dataset_type = config['dataset_kwargs'].get('dataset_type', '')
        if config['arch'] == 'cibhash':
            assert dataset_type == 'cibhash'
        train_dataset = configs.dataset(config,
                                        filename='train.txt',
                                        transform_mode='test' if no_augmentation else 'train',
                                        gpu_transform=gpu_transform,
                                        gpu_mean_transform=gpu_mean_transform,
                                        skip_preprocess=config['dataset_kwargs'].get('train_skip_preprocess', False))
        if dataset_type == 'rotation':
            logging.info('Using Rotation Dataset!')
            train_dataset = RotationDataset(train_dataset)
        elif dataset_type == 'simclr':
            logging.info('Using SimCLR Dataset!')
            train_dataset = InstanceDiscriminationDataset(train_dataset,
                                                          'simclr',
                                                          config['dataset_kwargs']['crop'],
                                                          config['dataset_kwargs']['weak_aug'])
        # lazy fix, can be more pretty and general, cibhash part 2/2
        elif dataset_type == 'cibhash':
            logging.info('Using CIBHash Dataset!')
            train_dataset = InstanceDiscriminationDataset(train_dataset,
                                                          'cibhash',
                                                          config['dataset_kwargs']['crop'],
                                                          weak_mode=0)
    else:
        train_dataset = None

    separate_multiclass = config['dataset_kwargs'].get('separate_multiclass', False)
    config['dataset_kwargs']['separate_multiclass'] = False
    return_id = config['dataset'] in ['landmark']

    test_dataset = configs.dataset(config,
                                   filename='test.txt',
                                   transform_mode='test',
                                   return_id=return_id,
                                   gpu_transform=gpu_transform,
                                   gpu_mean_transform=gpu_mean_transform,
                                   skip_preprocess=config['dataset_kwargs'].get('test_skip_preprocess', False))
    db_dataset = configs.dataset(config,
                                 filename='database.txt',
                                 transform_mode='test',
                                 return_id=return_id,
                                 gpu_transform=gpu_transform,
                                 gpu_mean_transform=gpu_mean_transform,
                                 skip_preprocess=config['dataset_kwargs'].get('db_skip_preprocess', False))

    config['dataset_kwargs']['separate_multiclass'] = separate_multiclass  # during mAP, no need to separate

    if train_dataset is not None:
        if train_full:
            logging.info('Loading full training dataset! Please be aware of memory usage')
            batch_size = len(train_dataset)
        else:
            batch_size = config['batch_size']

        train_loader = configs.dataloader(train_dataset,
                                          batch_size,
                                          shuffle=train_shuffle,
                                          drop_last=train_drop_last,
                                          workers=workers,
                                          seed=seed)
    else:
        train_loader = None

    maxbs = config.get('max_batch_size', 256)
    test_loader = configs.dataloader(test_dataset,
                                     maxbs,
                                     shuffle=test_shuffle,
                                     drop_last=False,
                                     workers=workers,
                                     seed=seed)
    db_loader = configs.dataloader(db_dataset,
                                   maxbs,
                                   shuffle=test_shuffle,
                                   drop_last=False,
                                   workers=workers,
                                   seed=seed)

    return train_loader, test_loader, db_loader


def prepare_model(config, device: torch.device):
    """

    :param config:
    {
        'arch': '',
        'arch_kwargs': {
            'arch': '',
            'nbit': 64,
            'nclass': 100,
            'pretrained': True,
            'freeze_weight': True,
            'bias': False,
            'backbone': 'alexnet',

            # linear
            'in_channels': 4096
        }
    }
    :param device:
    :return:
    """
    logging.info('Creating Model')
    model = configs.arch(config)
    # if (torch.cuda.device_count() > 1) and (config['device'] == 'cuda'):  # cuda device is not specified, use all
    #     logging.info('Using DataParallel Model')
    #     model = DataParallelPassthrough(model)

    if (torch.cuda.device_count() == 0) or (device.type == 'cpu'):  # cpu
        device_ids = []
        is_cpu = True
    elif (torch.cuda.device_count() > 0) and device.index is not None:  # select gpu
        device_ids = [device.index]
        is_cpu = False
    else:  # all gpu
        device_ids = None
        is_cpu = False
    logging.info(f'Using DataParallel Model. Device id = {device_ids}. is_cpu={is_cpu} device={device}')
    model = DataParallelPassthrough(model, device_ids=device_ids, is_cpu=is_cpu)

    model = model.to(device)
    return model
