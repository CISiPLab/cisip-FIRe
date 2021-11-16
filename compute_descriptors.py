import argparse
import os

import torch

import configs
from functions.loss.pca import PCALoss
from models import get_backbone
from scripts.train_helper import prepare_dataloader

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='cifar10', help='dataset')
parser.add_argument('--c10-ep', default=1, type=int)
parser.add_argument('--backbone', default='alexnet', help='backbone')
parser.add_argument('--savedir', required=True, help='save to')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--bs', default=64, type=int)
parser.add_argument('--pca-output', default=False, action='store_true')
parser.add_argument('--pca-dim', default=128, type=int)

args = parser.parse_args()

proceed = True

if os.path.exists(args.savedir):
    proceed = False

if not proceed:
    inp = input('Folder exists. Proceed and overwrite? (y/n)')
    if inp == 'y':
        proceed = True
    else:
        exit()

dataset_config = {
    'arch': '',
    'batch_size': args.bs,
    'max_batch_size': args.bs,
    'dataset': args.ds,
    'dataset_kwargs': {
        'no_augmentation': True,  # turn off augmentation
        'resize': configs.imagesize(args.ds),
        'crop': configs.cropsize(args.ds),
        'norm': 2,
        'evaluation_protocol': args.c10_ep
    },
    'arch_kwargs': {
        'nclass': configs.nclass(args.ds)
    }
}

train_loader, test_loader, db_loader = prepare_dataloader(dataset_config,
                                                          train_shuffle=False,
                                                          test_shuffle=False,
                                                          gpu_transform=False,
                                                          gpu_mean_transform=False,
                                                          include_train=True,
                                                          train_drop_last=False,
                                                          workers=os.cpu_count())

data_structure = {
    'train.txt': {
        'codes': [],
        'labels': []
    },
    'test.txt': {
        'codes': [],
        'labels': []
    },
    'database.txt': {
        'codes': [],
        'labels': []
    }
}

loaders = {
    'train.txt': train_loader,
    'test.txt': test_loader,
    'database.txt': db_loader,
}

backbone = get_backbone(backbone=args.backbone,
                        nbit=64,  # nbit and nclass will be ignored
                        nclass=configs.nclass(args.ds),
                        pretrained=True,
                        freeze_weight=True)
backbone.eval()

print(backbone)

device = torch.device(args.device)

backbone.to(device)

if args.pca_output:
    print('PCA enabled')
    pca = PCALoss(args.pca_dim)
else:
    pca = None

for filename in loaders:
    print(f'Filename: {filename}')
    loader = loaders[filename]

    for i, (data, labels, index) in enumerate(loader):
        print(f'Computing [{i}/{len(loader)}]', end='\r')
        data = data.to(device)

        with torch.no_grad():
            codes = backbone(data)

        if pca is not None and filename != 'train.txt':
            pca.eval()
            codes = pca(codes)

        data_structure[filename]['codes'].append(codes.cpu())
        data_structure[filename]['labels'].append(labels)

    print()
    data_structure[filename]['codes'] = torch.cat(data_structure[filename]['codes'])
    data_structure[filename]['labels'] = torch.cat(data_structure[filename]['labels'])

    if pca is not None and filename == 'train.txt':
        print('PCA training')
        pca.train()
        data_structure[filename]['codes'] = pca(data_structure[filename]['codes'])[0]

    print(f'Total number of data: {len(data_structure[filename]["codes"])}')

os.makedirs(args.savedir, exist_ok=True)

for filename in data_structure:
    saveto = args.savedir + '/' + filename
    torch.save(data_structure[filename], saveto)
    fsize = os.stat(saveto).st_size / (1024 * 1024)  # bytes -> Mbytes
    print(saveto)
    print(f'Filesize: {fsize:.4f} MB')
