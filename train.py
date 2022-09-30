import argparse
import ast
import collections.abc
import json
import logging
import os

import torch
import yaml
from pytorch_memlab import MemReporter

import configs
import constants
from scripts import train_general
from utils.logger import setup_logging
from utils.misc import dot_dict


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_loss_param_v2(loss_name, multiclass, nbit, device, loss_params_str: str):
    yaml_loss_param.update({
        'multiclass': multiclass,
        'nbit': nbit,
        'device': device
    })
    if loss_params_str:
        # parse and update/override
        loss_params = loss_params_str.replace(' ', '').split(';')
        for loss_param_str in loss_params:
            splits = loss_param_str.split(':')
            assert len(splits) == 2, 'Parsing Error for loss params'
            param, value = splits
            if param in yaml_loss_param:
                # The string or node provided may only consist of the following Python literal structures:
                # strings, bytes, numbers, tuples, lists, dicts, sets, booleans, and None.
                try:
                    value_type = type(ast.literal_eval(value))
                except ValueError:
                    value_type = str
                except SyntaxError:
                    value_type = str
                yaml_loss_param[param] = value_type(value)
            else:
                yaml_loss_param[param] = value
    return yaml_loss_param


def get_hash_layer(loss_name):
    if loss_name in ['greedyhash-unsupervised', 'greedyhash', 'cibhash']:
        return 'signhash'
    elif loss_name in ['jmlh', 'tbh']:
        return 'stochasticbin'
    elif loss_name in ['ssdh']:
        return 'tanh'
    else:
        return 'identity'


def get_arch(loss_name, arch_arg):
    if loss_name not in constants.supported_model:
        raise NotImplementedError(f'no implementation for {loss_name}')
    if arch_arg == '':
        model = constants.supported_model[loss_name][0]  # return default case
    elif arch_arg in constants.supported_model[loss_name]:
        model = arch_arg
    else:
        raise NotImplementedError(f'no implementation of {arch_arg} for {loss_name}.'
                                  f' Supported arch are: {constants.supported_model[loss_name]}')
    logging.info(f'Using architecture {arch_arg} for {loss_name} loss')
    return model


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    try:
        configs.default_workers = os.cpu_count() // torch.cuda.device_count()  # follow PyTorch recommendation
    except:
        # when running with no gpu, torch.cuda.device_count() = 0
        configs.default_workers = os.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='configs/templates/orthocos.yaml')
    parser.add_argument('--backbone', default='alexnet', type=str, help='the backbone feature extractor')
    parser.add_argument('--ds', default='imagenet100', choices=[dataset for key in constants.datasets
                                                                for dataset in constants.datasets[key]], help='dataset')
    parser.add_argument('--dfolder', default='', help='data folder')
    parser.add_argument('--c10-ep', default=1, type=int, choices=[1, 2], help='cifar10 evaluation protocol')
    parser.add_argument('--ds-reset', default=False, action='store_true', help='whether to reset cifar10 txt')
    parser.add_argument('--usedb', default=False, action='store_true', help='make all database images as training data')
    parser.add_argument('--train-ratio', default=1, type=float, help='training ratio (useful when usedb is activated)')
    parser.add_argument('--nbit', default=64, type=int, help='number of bits for hash codes')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--maxbs', default=256, type=int, help='maximum batch size for testing, by default it is max(bs * 4, maxbs)')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--arch', default='', type=str, help='architecture for the hash function')
    parser.add_argument('--gpu-transform', default=False, action='store_true')
    parser.add_argument('--gpu-mean-transform', default=False, action='store_true')
    parser.add_argument('--no-aug', default=False, action='store_true', help='whether to skip augmentation')
    parser.add_argument('--resize-size', default=-1, type=int, help='Image Resize size before crop')
    parser.add_argument('--crop-size', default=-1, type=int, help='Image Crop size. Final image size.')
    parser.add_argument('--R', default=0, type=int, help='if 0, using default R for specific dataset; -1 for mAP@All')
    parser.add_argument('--distance-func', default='hamming', choices=['hamming', 'cosine', 'euclidean'])
    parser.add_argument('--zero-mean-eval', default=False, action='store_true')
    parser.add_argument('--num-worker', default=-1, type=int, help='number of worker for dataloader')
    parser.add_argument('--rand-aug', default=False, action='store_true', help='use random augmentation')
    # change: only define at losses
    parser.add_argument('--loss', default='dpn', choices=[name for loss in constants.losses
                                                          for name in constants.losses[loss]])
    parser.add_argument('--tag', default='test')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop', 'adan'])
    parser.add_argument('--loss-params', default='', type=str)

    parser.add_argument('--device', default='cuda:0', type=str, help='torch.device(\'?\') cpu, cuda:x')
    parser.add_argument('--eval', default=10, type=int, help='total evaluations throughout the training')

    # lr related
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--step-size', default=0.8, type=float, help='relative step size (0~1)')
    parser.add_argument('--lr-decay-rate', default=0.1, type=float, help='decay rate for lr')
    parser.add_argument('--scheduler', default='step', type=str, help='LR Scheduler')
    parser.add_argument('--backbone-lr-scale', default=0.1, type=float, help='Scale the learning rate of CNN backbone')
    """Resume Training
        ``` --resume --resume-dir <logdir to resume> ```
    """
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume-dir', default='', type=str, help='resume dir')
    parser.add_argument('--enable-checkpoint', default=False, action='store_true')
    parser.add_argument('--save-model', default=False, action='store_true')
    parser.add_argument('--save-best-model-only', default=False, action='store_true')
    parser.add_argument('--discard-hash-outputs', default=False, action='store_true')
    parser.add_argument('--load-from', default='', type=str, help='whether to load from a model')
    parser.add_argument('--benchmark', default=False, action='store_true',
                        help='Benchmark mode, determinitic, and no loss')
    parser.add_argument('--disable-tqdm', default=False, action='store_true', help='disable tqdm for less verbose stderr')

    parser.add_argument('--hash-bias', default=False, action='store_true', help='add bias to hash_fc')

    # evaluation
    parser.add_argument('--shuffle-database', default=False, action='store_true',
                        help='shuffle database during mAP evaluation')

    parser.add_argument('--workers', default=-1, type=int, help='number of workers')
    parser.add_argument('--train-skip-preprocess', default=False, action='store_true')
    parser.add_argument('--db-skip-preprocess', default=False, action='store_true')
    parser.add_argument('--test-skip-preprocess', default=False, action='store_true')
    parser.add_argument('--dataset-name-suffix', default='')

    # image backend
    parser.add_argument('--accimage', default=False, action='store_true', help='use accimage as backend')
    parser.add_argument('--pin-memory', default=False, action='store_true', help='pin memory')

    # wandb settings
    parser.add_argument('--wandb', action='store_true', default=False, help='enable wandb logging')

    args = parser.parse_args()
    yaml_loss_param = {}
    custom_param = {}

    if args.config:
        print("Using yaml, args have higher priority, loading")
        # args priority is higher than yaml
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        types = {}
        opt = vars(args)
        for arg in opt:
            if type(opt[arg]) is bool:
                aux_parser.add_argument('--' + arg.replace('_', '-'), action='store_true')
            else:
                aux_parser.add_argument('--' + arg.replace('_', '-'))
            types[arg] = type(opt[arg])
        cli_args, _ = aux_parser.parse_known_args()

        data = yaml.load(open(args.config), Loader=yaml.FullLoader)
        if 'loss_param' in data:
            yaml_loss_param = data['loss_param']
        if 'custom_param' in data:
            custom_param = data['custom_param']
        opt.update(data)

        for arg in cli_args.__dict__:
            try:
                opt[arg] = types[arg](cli_args.__dict__[arg])
            except TypeError:
                logging.error(f"Argument {arg} incorrectly parsed.")

        args = dot_dict(opt)
    else:
        print("Using only args.")

    if args.workers != -1:
        configs.default_workers = min(args.workers, configs.default_workers)

    if args.accimage:
        logging.info('Using accimage as backend!')
        configs.use_accimage_backend()

    if args.pin_memory:
        configs.pin_memory = True

    if args.ds == 'descriptor':
        assert args.dfolder != '', 'please input --dfolder'
        ds = '_'.join(args.dfolder.split('_')[:-1])
    else:
        ds = args.ds

    arch = get_arch(args.loss, args.arch)
    nbit = args.nbit
    nclass = configs.nclass(ds)  # just a dummy one, will be reset below
    lr = args.lr
    epochs = args.epochs
    tag = args.tag
    if args.seed == -1:
        seed = torch.randint(100000, ()).item()
    else:
        seed = args.seed

    # pre-loading data folder
    dfolder = args.dfolder
    if 'cifar10' in dfolder:
        dfolder = dfolder + '_' + str(args.c10_ep)

    if dfolder == '':
        data_folder = ''
    else:
        data_folder = constants.descriptors_data_folder[dfolder]

    config = {
        'arch': arch,
        'arch_kwargs': {
            'arch': arch,
            'nbit': nbit,
            'nclass': nclass,
            'pretrained': True,
            'freeze_weight': True if args.backbone_lr_scale == 0 else False,
            'bias': args.hash_bias,
            'backbone': args.backbone,

            # linear
            'in_channels': configs.in_features(dfolder, dataset=args.ds),
        },
        'batch_size': args.bs,
        'max_batch_size': args.maxbs,
        'dataset': args.ds,  # notice that it is using args.ds instead of ds
        'device': args.device,
        'dfolder': args.dfolder,
        'multiclass': ds in constants.datasets['multiclass'],
        'dataset_kwargs': {
            'resize': configs.imagesize(ds) if args.resize_size == -1 else args.resize_size,
            'crop': configs.cropsize(ds) if args.crop_size == -1 else args.crop_size,
            'norm': 2,
            'evaluation_protocol': args.c10_ep,
            'use_db_as_train': args.usedb,  # for 50a and 50b
            'train_ratio': args.train_ratio,
            'reset': args.ds_reset,
            'separate_multiclass': False,
            'train_skip_preprocess': args.train_skip_preprocess,
            'db_skip_preprocess': args.db_skip_preprocess,
            'test_skip_preprocess': args.test_skip_preprocess,
            'dataset_name_suffix': args.dataset_name_suffix,  # e.g. "_resize", it will load "data/xxx_resize"
            'neighbour_topk': 5,  # for neighbour dataset
            'no_augmentation': args.no_aug,
            'data_folder': data_folder,
            'use_random_augmentation': args.rand_aug
        },
        'optim': args.optim,
        'optim_kwargs': {
            'lr': lr,
            'momentum': 0.9,  # sgd, rms
            'nesterov': False,  # sgd
            'betas': (0.9, 0.999),  # adam
            'alpha': 0.99,  # rms
            'weight_decay': args.wd,
        },
        'epochs': epochs,
        'scheduler': args.scheduler,
        'scheduler_kwargs': {
            'step_size': max(1, int(args.epochs * args.step_size)),  # get_stepsize(args.loss),
            'gamma': args.lr_decay_rate,
            'milestones': '0.5,0.75',
            'linear_init_lr': 0.001,
            'linear_last_lr': 0.00001,
        },
        'save_interval': 0,
        'eval_interval': (epochs // args.eval) if args.eval else 0,
        'shuffle_database': args.shuffle_database,
        'tag': tag,
        'seed': seed,
        'loss': args.loss,
        'loss_param': get_loss_param_v2(args.loss,
                                        multiclass=ds in constants.datasets['multiclass'],
                                        nbit=nbit,
                                        device=args.device, loss_params_str=args.loss_params),
        'backbone_lr_scale': args.backbone_lr_scale,
        'start_epoch_from': 0,
        'resume_dir': args.resume_dir,
        'save_checkpoint': args.enable_checkpoint,
        'load_from': args.load_from,
        'save_model': args.save_model,
        'save_best_model_only': args.save_best_model_only,
        'discard_hash_outputs': args.discard_hash_outputs,
        'benchmark': args.benchmark,
        'num_worker': args.num_worker,
        'wandb_enable': args.wandb,
        'disable_tqdm': args.disable_tqdm
    }

    if args.R == 0:
        config['dataset'] = ds  # using original dataset for convenient
        config['R'] = configs.R(config)
        config['dataset'] = args.ds  # it switch back to descriptor, if it is descriptor
    else:
        config['R'] = args.R

    if args.distance_func == 'hamming':
        config['distance_func'] = 'jmlh-dist' if config['loss'] in ['jmlh', 'tbh'] else 'hamming'
    else:
        config['distance_func'] = args.distance_func

    config['zero_mean_eval'] = args.zero_mean_eval
    if len(custom_param) != 0:
        logging.info('Custom Param enabled! No overridden will be perform')
        update(config, custom_param)

    configs.disable_tqdm = config['disable_tqdm']

    if args.resume:
        resume_dir = args.resume_dir
        logging.info(f"resume from {resume_dir}")
        config = json.load(open(resume_dir + "/config.json"))
        train_history = json.load(open(resume_dir + "/train_history.json"))
        last_epoch = len(train_history)
        config['start_epoch_from'] = train_history[-1]['ep']
        config['save_checkpoint'] = args.enable_checkpoint
        config['resume_dir'] = resume_dir
        logdir = resume_dir
    else:
        if args.ds == 'descriptor':
            ds_prefix = dfolder
        else:
            ds_prefix = ds
        logdir = (
            f'logs/{args.loss}_{config["arch_kwargs"]["backbone"]}_{config["arch"]}_{config["arch_kwargs"]["nbit"]}_'
            f'{ds_prefix}_'
            f'{config["epochs"]}_'
            f'{config["optim_kwargs"]["lr"]}_'
            f'{config["optim"]}')

        # always ensure path name is "logdir/count_tag_seed"
        latest_count = -1

        if os.path.isdir(logdir):
            for dirname in os.listdir(logdir):
                try:
                    count = int(dirname.split('_')[0])
                    latest_count = max(count, latest_count)
                except ValueError:
                    print(dirname)

        latest_count += 1

        if config['tag'] != '':
            logdir += f'/{latest_count:03d}_{config["tag"]}_{config["seed"]}'
        else:
            logdir += f'/{latest_count:03d}_{config["seed"]}'

        config['logdir'] = logdir
        os.makedirs(logdir, exist_ok=True)

    # setup logger
    setup_logging(logdir + '/log.txt')

    logging.info(f'Log directory: {logdir}')

    try:
        method = None
        for key in constants.losses:
            if args.loss in constants.losses[key]:
                method = key

        if method is None:
            raise NotImplementedError(f"Loss {args.loss} not found in {list(constants.losses.keys())}")

        gpu_mean_transform = args.gpu_mean_transform
        if args.loss in ['cibhash']:
            config['dataset_kwargs']['no_augmentation'] = True
            config['dataset_kwargs']['train_skip_preprocess'] = True
            config['dataset_kwargs']['dataset_type'] = 'cibhash'
            gpu_mean_transform = True
            logging.info('GPU Mean Transform enabled.')
        train_general.main(config,
                           gpu_transform=False,
                           gpu_mean_transform=gpu_mean_transform,
                           method=method)
    except RuntimeError as e:
        reporter = MemReporter()
        reporter.report(verbose=True)
        raise e
