import gc
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from tqdm import tqdm
import wandb

import configs
from functions.hashing import get_hadamard
from functions.metrics import calculate_mAP
from scripts import (
    train_helper,
    train_supervised,
    train_unsupervised,
    train_pairwise,
    train_autoencoder,
    train_adversarial,
    train_shallow,
    train_contrastive
)
from scripts.train_helper import (
    prepare_dataloader,
    prepare_model,
    get_loss,
    generate_centroids,
    get_dataloader)
from utils import io
from utils.augmentations import (
    get_mean_transforms_gpu
)
from utils.misc import Timer, AverageMeter


def backward_general(output, loss, optimizer,
                     method, model, criterion,
                     data, labels, index,
                     onehot, stage, loss_name, loss_cfg):
    if method == 'adversarial':
        f = train_adversarial.optimize_params_adv
        f(output, loss, optimizer,
          method, model, criterion,
          data, labels, index,
          onehot, stage, loss_name, loss_cfg)
    else:
        loss.backward()
        optimizer.step()


def get_output_and_loss_general(method, model, criterion,
                                data, labels, index,
                                onehot, stage, loss_name, loss_cfg, no_loss) -> Tuple[Dict, Any]:
    f = {
        'supervised': train_supervised.get_output_and_loss_supervised,
        'unsupervised': train_unsupervised.get_output_and_loss_unsupervised,
        'pairwise': train_pairwise.get_output_and_loss_pairwise,
        'autoencoder': train_autoencoder.get_output_and_loss_autoencoder,
        'adversarial': train_adversarial.get_output_and_loss_adv,
        'contrastive': train_contrastive.get_output_and_loss_contrastive
    }.get(method)

    if f is None:
        raise NotImplementedError(f'Method {method} not implemented for get_output_and_loss_general')

    return f(model, criterion, data, labels, index, onehot, loss_name, loss_cfg, stage, no_loss)


def update_meters_general(method, model, meters, out, labels, onehot, criterion, loss_name, loss_cfg):
    f = {
        'supervised': train_supervised.update_meters_supervised,
        'unsupervised': train_unsupervised.update_meters_unsupervised,
        'pairwise': train_pairwise.update_meters_pairwise,
        'autoencoder': train_autoencoder.update_meters_autoencoder,
        'adversarial': train_adversarial.update_meters_adv,
        'contrastive': train_contrastive.update_meters_contrastive
    }.get(method)

    if f is None:
        raise NotImplementedError(f'Method {method} not implemented for update_meters_general')

    return f(model, meters, out, labels, onehot, criterion, loss_name, loss_cfg)


def update_loss_params(method, loss_param, train_loader, nbit, nclass):
    if method == 'supervised':
        pass
    elif method == 'unsupervised':
        pass
    elif method == 'pairwise':
        return train_pairwise.update_params_pairwise(loss_param, train_loader, nbit, nclass)
    elif method == 'autoencoder':
        pass
    elif method == 'adversarial':
        pass
    elif method == 'contrastive':
        pass
    else:
        raise NotImplementedError(f'Method {method} not implemented for update_loss_params')


def prepare_dataset_from_model(model, criterion, config, train_loader, test_loader, db_loader):
    if config['loss'] in ['ssdh', 'uhbdnn']:
        criterion.prepare_dataset_from_model(model, config,
                                             train_loader,
                                             test_loader,
                                             db_loader)


def pre_epoch_operations(loss, **kwargs):
    if loss == 'hashnet':
        ep = kwargs['ep']
        loss_param = kwargs['loss_param']
        step_continuation = loss_param['loss_param']['step_continuation']
        kwargs['criterion'].beta = (ep // step_continuation + 1) ** 0.5
        logging.info(f'updated scale: {kwargs["criterion"].beta}')


def train_hashing(optimizer, model, train_loader, device, loss_name, loss_cfg, onehot,
                  gpu_train_transform=None, method='supervised', criterion=None, logdir=None):
    model.train()

    batch_timer = Timer()
    total_timer = Timer()
    total_timer.tick()
    if criterion is None:
        criterion = train_helper.get_loss(loss_name, **loss_cfg)
    meters = defaultdict(AverageMeter)

    train_helper.update_criterion(model=model, criterion=criterion, loss_name=loss_name, method=method, onehot=onehot)
    criterion.train()

    pbar = tqdm(train_loader, desc='Train', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)
    batch_timer.tick()

    running_times = []

    for i, batch in enumerate(pbar):
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.zero_grad()
        else:
            optimizer.zero_grad()

        data, labels, index = batch[:3]
        data, labels = data.to(device), labels.to(device)
        if gpu_train_transform is not None:
            if len(data.shape) == 5:
                data[:, 0], data[:, 1] = gpu_train_transform(data[:, 0]), gpu_train_transform(data[:, 1])
            else:
                data = gpu_train_transform(data)

        if len(batch) == 6:  # it is a neighbor dataset
            ndata, nlabels, nindex = batch[3:]
            if gpu_train_transform is not None:
                ndata = gpu_train_transform(ndata)
            data = (data, ndata)
            labels = (labels, nlabels)
            index = (index, nindex)

        output, loss = get_output_and_loss_general(method, model, criterion,
                                                   data, labels, index,
                                                   onehot, 'train', loss_name, loss_cfg, no_loss=False)
        backward_general(output, loss, optimizer,
                         method, model, criterion,
                         data, labels, index,
                         onehot, 'train', loss_name, loss_cfg)
        batch_timer.toc()

        update_meters_general(method, model, meters, output, labels, onehot, criterion, loss_name, loss_cfg)
        criterion.losses.clear()  # clear the key in losses

        meters['loss'].update(loss.item())
        meters['time'].update(batch_timer.total)
        running_times.append(batch_timer.total)
        pbar.set_postfix({key: val.avg for key, val in meters.items()})
        batch_timer.tick()

        # if i % 2 == 0:
        #     io.fast_save(output['code_logits'].detach().cpu(), f'{logdir}/outputs/train_iter_{i}.pth')
        # if i > 200:
        #     import sys
        #     sys.exit(0)

    total_timer.toc()
    meters['total_time'].update(total_timer.total)
    std_time = f"time_std={np.std(running_times[1:]):.5f}"
    mean_time = f"time_mean={np.mean(running_times[1:]):.5f}"
    logging.info(f'{"; ".join([f"{key}={val.avg:.4f}" for key, val in meters.items()] + [mean_time, std_time])}')

    return meters


def test_hashing(model, test_loader, device, loss_name, loss_cfg, onehot, return_codes=False,
                 return_id=False, gpu_test_transform=None, method='supervised', criterion=None, no_loss=False):
    model.eval()

    total_timer = Timer()
    batchtimer = Timer()
    total_timer.tick()

    ret_codes = []
    ret_labels = []
    ret_id = []
    meters = defaultdict(AverageMeter)
    if criterion is None:
        criterion = train_helper.get_loss(loss_name, **loss_cfg)

    train_helper.update_criterion(model=model, criterion=criterion, loss_name=loss_name, method=method, onehot=onehot)
    criterion.eval()

    pbar = tqdm(test_loader, desc='Test', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)
    batchtimer.tick()

    running_times = []

    for i, batch in enumerate(pbar):
        with torch.no_grad():
            data, labels, index = batch[:3]
            data, labels = data.to(device), labels.to(device)

            if gpu_test_transform is not None:
                data = gpu_test_transform(data)

            if len(batch) == 6:  # it is a neighbor dataset
                ndata, nlabels, nindex = batch[3:]
                if gpu_test_transform is not None:
                    ndata = gpu_test_transform(ndata)
                data = (data, ndata)
                labels = (labels, nlabels)
                index = (index, nindex)
            output, loss = get_output_and_loss_general(method, model, criterion,
                                                       data, labels, index,
                                                       onehot, 'test', loss_name, loss_cfg, no_loss)
            batchtimer.toc()
            running_times.append(batchtimer.total)

            if return_codes:
                ret_codes.append(output['code_logits'].cpu())
                ret_labels.append(labels)
                if return_id:
                    ret_id.append(index[0])

        update_meters_general(method, model, meters, output, labels, onehot, criterion, loss_name, loss_cfg)
        criterion.losses.clear()  # clear the key in losses

        meters['loss'].update(loss.item())
        meters['time'].update(batchtimer.total)
        pbar.set_postfix({key: val.avg for key, val in meters.items()})
        batchtimer.tick()
    total_timer.toc()
    meters['total_time'].update(total_timer.total)
    std_time = f"time_std={np.std(running_times[1:]):.5f}"
    mean_time = f"time_mean={np.mean(running_times[1:]):.5f}"
    logging.info(f'{"; ".join([f"{key}={val.avg:.4f}" for key, val in meters.items()] + [mean_time, std_time])}')
    if return_codes:
        res = {
            'codes': torch.cat(ret_codes),
            'labels': torch.cat(ret_labels),
            'id': np.concatenate(ret_id) if len(ret_id) else np.array([])
        }
        return meters, res
    return meters


def resume_training(config, logdir,
                    model, optimizer, scheduler,
                    criterion, train_history, test_history):
    if config['start_epoch_from'] != 0:
        assert config['loss'] not in ['dpn', 'csq'], 'centroid restore not implemented yet'
        ckpt = torch.load(logdir + '/checkpoint.pth')
        model.load_state_dict(ckpt['model'])
        if isinstance(optimizer, list):
            optim_sd = ckpt['optimizer']
            scheduler_sd = ckpt['scheduler']
            assert len(optim_sd) == len(optimizer), 'the length of optim_sd != optimizer, please check'
            assert len(scheduler_sd) == len(scheduler), 'the length of scheduler_sd != scheduler, please check'
            assert len(scheduler) == len(optimizer), 'the length of scheduler != optimizer, please check'
            for sch_sd, opt_sd, sch, opt in zip(scheduler_sd, optim_sd, scheduler, optimizer):
                opt.load_state_dict(opt_sd)
                sch.load_state_dict(sch_sd)
        else:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        criterion = ckpt['criterion']
        assert ckpt['epoch'] + 1 == config['start_epoch_from']
        train_history = json.load(open(config['resume_dir'] + "/train_history.json"))
        test_history = json.load(open(config['resume_dir'] + "/test_history.json"))

        logging.info(f'Resume Training from epoch {ckpt["epoch"] + 1}....')

    return criterion, train_history, test_history


def load_gpu_transform(config, gpu_transform, gpu_mean_transform):
    # if gpu_transform:
    #     logging.info("Loading gpu transform")
    #     gpu_train_transform = get_transforms_gpu(config['dataset'], 'train',
    #                                              resize=config['dataset_kwargs'].get('resize', 0),
    #                                              crop=config['dataset_kwargs'].get('crop', 0))
    #     gpu_test_transform = get_transforms_gpu(config['dataset'], 'test',
    #                                             resize=config['dataset_kwargs'].get('resize', 0),
    #                                             crop=config['dataset_kwargs'].get('crop', 0))
    if gpu_mean_transform:
        gpu_train_transform = gpu_test_transform = get_mean_transforms_gpu(norm=2)
    else:
        gpu_train_transform = None
        gpu_test_transform = None

    return gpu_train_transform, gpu_test_transform


def preprocess(model, config, device):
    centroids = None

    if config['loss'] in ['dpn', 'orthocos', 'orthoarc']:
        init_method = config['loss_param']['cent_init']
        nclass, nbit = config['arch_kwargs']['nclass'], config['arch_kwargs']['nbit']
        centroids = generate_centroids(nclass, nbit, init_method)
        centroids = centroids.to(device)
        centroids = centroids.sign()

        model.centroids = centroids
        if config['loss'] in ['orthoarc', 'orthocos']:
            model.ce_fc.centroids.data.copy_(centroids)

    elif config['loss'] in ['csq']:
        #  https://github.com/swuxyj/DeepHash-pytorch/blob/master/CSQ.py
        logging.info('Preprocessing for CSQ')
        nclass = config['arch_kwargs']['nclass']
        nbit = config['arch_kwargs']['nbit']
        # centroids = get_hadamard(nclass, nbit, fast=True)
        centroids = generate_centroids(nclass, nbit, 'B')
        logging.info("using bernoulli")
        centroids = centroids.to(device)

        # move to model
        model.centroids = centroids

    logdir = config['logdir']
    if centroids is not None:
        io.fast_save(centroids.clone().cpu(), f'{logdir}/outputs/centroids.pth')
    logging.info('Preprocessing done!')


def load_model(model, config):
    if config['load_from'] != '':
        logging.info(f'Loading from {config["load_from"]}')
        sd = torch.load(config['load_from'], map_location='cpu')
        msg = model.load_state_dict(sd, strict=False)
        logging.info(f'{msg}')


def main(config, gpu_transform=False, gpu_mean_transform=False, method='supervised'):
    benchmark = config['benchmark']
    if benchmark:
        cudnn.deterministic = True
    ##### experiment initialization #####
    device = torch.device(config['device'])
    io.init_save_queue()

    total_timer = Timer()
    total_timer.tick()
    configs.seeding(config['seed'])

    logdir = config['logdir']
    assert logdir != '', 'please input logdir'

    if config['wandb_enable']:
        ## initiaze wandb ##
        wandb_dir = logdir
        wandb.init(config=config, dir=wandb_dir)
        # wandb run name
        wandb.run.name = logdir.split('logs/')[1]

    logging.info(json.dumps(config, indent=2))

    os.makedirs(f'{logdir}/models', exist_ok=True)
    os.makedirs(f'{logdir}/optims', exist_ok=True)
    os.makedirs(f'{logdir}/outputs', exist_ok=True)
    json.dump(config, open(f'{logdir}/config.json', 'w+'), indent=4, sort_keys=True)

    ##### go to shallow #####
    if method == 'shallow':
        best = train_shallow.main(config)
        total_timer.toc()
        total_time = total_timer.total
        io.join_save_queue()
        logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
        logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
        logging.info(f'Best mAP: {best:.6f}')
        logging.info(f'Done: {logdir}')
        return

    ##### dataset preparation #####
    workers = 1 if config['dataset'] in ['gldv2delgembed'] else config['num_worker']
    train_loader, test_loader, db_loader = prepare_dataloader(config,
                                                              gpu_transform=gpu_transform,
                                                              gpu_mean_transform=gpu_mean_transform,
                                                              workers=workers, seed=config['seed'])

    ##### model preparation #####
    model = prepare_model(config, device)
    logging.info(model)
    load_model(model, config)
    preprocess(model, config, device)

    if config['wandb_enable']:
        wandb.watch(model)

    ##### pre-training initilization #####
    loss_param = config.copy()
    loss_param.update({
        'device': device
    })
    nbit = config['arch_kwargs']['nbit']
    nclass = config['arch_kwargs']['nclass']
    update_loss_params(method=method, loss_param=loss_param, train_loader=train_loader, nbit=nbit, nclass=nclass)
    criterion = get_loss(loss_param['loss'], **loss_param['loss_param'])
    criterion = criterion.to(device)

    ##### optimizer and scheduler preparation #####
    if config['backbone_lr_scale'] != 0:
        param_groups = [
            {'params': model.get_features_params(), 'lr': config['optim_kwargs']['lr'] * config['backbone_lr_scale']},
            {'params': model.get_hash_params()},
            {'params': criterion.parameters()}
        ]
    else:
        param_groups = [
            {'params': model.get_hash_params()},
            {'params': criterion.parameters()}
        ]

    optimizer = configs.optimizer(config, param_groups)
    scheduler = configs.scheduler(config, optimizer)

    if method == 'adversarial':
        param_groups = [
            {'params': model.get_discriminator_params()}
        ]
        adv_opt = configs.optimizer(config, param_groups)
        optimizer = [optimizer, adv_opt]
        scheduler = [scheduler, configs.scheduler(config, adv_opt)]

    train_history = []
    test_history = []

    logging.info(f'Total Bit: {nbit}')
    if config['dataset'] == 'descriptor':
        dataset_name = config['dfolder']
    else:
        dataset_name = config['dataset']
    onehot = not (dataset_name in configs.non_onehot_dataset)
    calculate_mAP_using_id = dataset_name in configs.dataset_evaluated_by_id

    if calculate_mAP_using_id:
        ground_truth_path = os.path.join(test_loader.dataset.root, 'ground_truth.csv')
        ground_truth = pd.read_csv(ground_truth_path)  # id = index id, images = images id in database

    ##### resume training #####
    if config['start_epoch_from'] != 0:
        criterion, train_history, test_history = resume_training(config, logdir,
                                                                 model, optimizer, scheduler,
                                                                 criterion, train_history, test_history)

    ##### gpu transform #####
    gpu_train_transform, gpu_test_transform = load_gpu_transform(config, gpu_transform, gpu_mean_transform)

    ##### start training #####
    logging.info('Training Start')

    best = 0
    curr_metric = 0
    nepochs = config['epochs']
    neval = config['eval_interval']

    ##### pre-training operations #####
    prepare_dataset_from_model(model, criterion, config,
                               train_loader, test_loader, db_loader)

    for ep in range(config['start_epoch_from'], nepochs):
        ##### clean up and print message #####
        if isinstance(scheduler, list):
            logging.info(f'Epoch [{ep + 1}/{nepochs}] LR: {scheduler[0].get_last_lr()[-1]:.6f}')
        else:
            logging.info(f'Epoch [{ep + 1}/{nepochs}] LR: {scheduler.get_last_lr()[-1]:.6f}')
        res = {'ep': ep + 1}
        gc.collect()
        # torch.cuda.empty_cache()

        ##### pre-epoch operations #####
        pre_epoch_operations(loss_param['loss'], loss_param=loss_param, ep=ep,
                             config=config, model=model, criterion=criterion, loader=train_loader)

        ##### train for an epoch #####
        train_meters = train_hashing(optimizer, model, train_loader, device, loss_param['loss'],
                                     loss_param['loss_param'], onehot=onehot,
                                     gpu_train_transform=gpu_train_transform,
                                     method=method, criterion=criterion, logdir=logdir)

        ##### scheduler #####
        if isinstance(scheduler, list):
            for sch in scheduler:
                sch.step()
        else:
            scheduler.step()

        ##### store meters #####
        for key in train_meters: res['train_' + key] = train_meters[key].avg
        train_history.append(res)

        if config['wandb_enable']:
            wandb_train = res.copy()
            wandb_train.pop("ep")
            wandb.log(wandb_train, step=res['ep'])

        ##### model saving #####
        modelsd = model.state_dict()
        if config['save_checkpoint']:
            if isinstance(optimizer, list):
                optim_sd = [opt.state_dict() for opt in optimizer]
                scheduler_sd = [sch.state_dict() for sch in scheduler]
            else:
                optim_sd = optimizer.state_dict()
                scheduler_sd = scheduler.state_dict()

            io.fast_save({
                'model': modelsd,
                'optimizer': optim_sd,
                'scheduler': scheduler_sd,
                'criterion': criterion,
                'epoch': ep
            }, f'{logdir}/checkpoint.pth')

        ##### evaluation #####
        is_last = (ep + 1) == nepochs
        eval_now = is_last or (neval != 0 and (ep + 1) % neval == 0)
        if eval_now:
            res = {'ep': ep + 1}

            ##### obtain testing and database codes and statistics #####
            test_meters, test_out = test_hashing(model, test_loader, device, loss_param['loss'],
                                                 loss_param['loss_param'], onehot=onehot,
                                                 return_codes=True, return_id=calculate_mAP_using_id,
                                                 gpu_test_transform=gpu_test_transform, method=method,
                                                 criterion=criterion, no_loss=benchmark)
            db_meters, db_out = test_hashing(model, db_loader, device, loss_param['loss'],
                                             loss_param['loss_param'], onehot=onehot,
                                             return_codes=True, return_id=calculate_mAP_using_id,
                                             gpu_test_transform=gpu_test_transform, method=method,
                                             criterion=criterion, no_loss=benchmark)
            for key in test_meters: res['test_' + key] = test_meters[key].avg
            for key in db_meters: res['db_' + key] = db_meters[key].avg

            map_device = device
            if db_out['codes'].numel() > 400000000:
                db_out['codes'] = db_out['codes'].cpu()
                test_out['codes'] = test_out['codes'].cpu()
                map_device = torch.device('cpu')

            ##### compute mAP #####
            res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                                       test_out['codes'], test_out['labels'],
                                       loss_param['R'], device=map_device, onehot=onehot,
                                       using_id=calculate_mAP_using_id,
                                       ground_truth=ground_truth if calculate_mAP_using_id else None,
                                       db_id=db_out['id'] if calculate_mAP_using_id else None,
                                       test_id=test_out['id'] if calculate_mAP_using_id else None,
                                       shuffle_database=config['shuffle_database'],
                                       distance_func=config['distance_func'],
                                       zero_mean=config['zero_mean_eval'])
            logging.info(f'mAP: {res["mAP"]:.6f}')

            curr_metric = res['mAP']
            test_history.append(res)

            if config['wandb_enable']:
                wandb_test = res.copy()
                wandb_test.pop("ep")
                wandb.log(wandb_test, step=res['ep'])

            if not config['save_best_model_only']:
                io.fast_save(db_out, f'{logdir}/outputs/db_out.pth')
                io.fast_save(test_out, f'{logdir}/outputs/test_out.pth')
            if best < curr_metric:
                best = curr_metric
                if config['wandb_enable']:
                    wandb.run.summary["best_map"] = best
                if config['save_model']:
                    io.fast_save(modelsd, f'{logdir}/models/best.pth')
                    if not config['discard_hash_outputs']:
                        io.fast_save(db_out, f'{logdir}/outputs/db_best.pth')
                        io.fast_save(test_out, f'{logdir}/outputs/test_best.pth')
            del db_out, test_out

            ##### obtain training codes and statistics #####
            if is_last and not config['dataset'] in configs.embedding_datasets:  # embed dataset oom
                # reload for eval mode
                train_loader = get_dataloader(config,
                                              'train.txt',
                                              shuffle=False,
                                              drop_last=False,
                                              gpu_transform=gpu_transform,
                                              gpu_mean_transform=gpu_mean_transform,
                                              no_augmentation=True,
                                              skip_preprocess=False,  # do not skip as using test mode
                                              dataset_type='',
                                              full_batchsize=False,
                                              seed=config['seed'])
                _, train_out = test_hashing(model, train_loader, device, loss_param['loss'],
                                            loss_param['loss_param'], onehot=onehot,
                                            return_codes=True, return_id=calculate_mAP_using_id,
                                            gpu_test_transform=gpu_test_transform, method=method,
                                            criterion=criterion)

                io.fast_save(train_out, f'{logdir}/outputs/train_out.pth')

        ##### save model, codes and statistics #####
        json.dump(train_history, open(f'{logdir}/train_history.json', 'w+'), indent=2, sort_keys=True)

        if len(test_history) != 0:
            json.dump(test_history, open(f'{logdir}/test_history.json', 'w+'), indent=2, sort_keys=True)

        save_now = config['save_interval'] != 0 and (ep + 1) % config['save_interval'] == 0
        if save_now and config['save_model']:
            io.fast_save(modelsd, f'{logdir}/models/ep{ep + 1}.pth')

    ##### training end #####
    modelsd = model.state_dict()
    if config['save_model'] and not config['save_best_model_only']:
        io.fast_save(modelsd, f'{logdir}/models/last.pth')

    total_timer.toc()
    total_time = total_timer.total
    io.join_save_queue()
    logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info(f'Best mAP: {best:.6f}')
    logging.info(f'Done: {logdir}')

    return logdir


def update_bn_mean(model, db_loader, config, device, method):
    model.hash_fc[1].bias = torch.nn.Parameter(torch.zeros_like(model.hash_fc[1].bias))
    model.hash_fc[1].weight = torch.nn.Parameter(torch.ones_like(model.hash_fc[1].weight))
    model.hash_fc[1].running_mean = torch.zeros_like(model.hash_fc[1].running_mean)
    model.hash_fc[1].running_var = torch.ones_like(model.hash_fc[1].running_var)
    # get code
    _, db_out = test_hashing(model, db_loader, device, config['loss'],
                             config['loss_param'], return_codes=True,
                             onehot=False,
                             return_id=False, method=method)

    model.hash_fc[1].running_mean = db_out['codes'].mean(0).to(device)
    model.hash_fc[1].running_var = db_out['codes'].var(0).to(device)
