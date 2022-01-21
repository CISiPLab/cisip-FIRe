import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import configs
from functions.metrics import calculate_mAP
from scripts.train_helper import get_loss, prepare_dataloader
from utils import io


def obtain_codes(criterion, loader, device, return_id=False):
    criterion.eval()
    pbar = tqdm(loader, desc='Obtain Codes', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)

    ret_codes = []
    ret_labels = []
    ret_ids = []
    for i, batch in enumerate(pbar):
        data, labels, index = batch[:3]
        data = data.to(device)

        with torch.no_grad():
            codes = criterion(data)

            ret_codes.append(codes.cpu())
            ret_labels.append(labels)
            if return_id:
                ret_ids.append(index[0])  # (landmark_id, index) for index

    res = {
        'codes': torch.cat(ret_codes),
        'labels': torch.cat(ret_labels),
        'id': np.concatenate(ret_ids) if len(ret_ids) else np.array([])
    }
    return res


def main(config):
    device = config['device']

    nbit = config['arch_kwargs']['nbit']
    logging.info(f'Total Bit: {nbit}')
    if config['dataset'] == 'descriptor':
        dataset_name = config['dfolder']
    else:
        dataset_name = config['dataset']
    onehot = not (dataset_name in configs.non_onehot_dataset)
    calculate_mAP_using_id = dataset_name in configs.dataset_evaluated_by_id

    ##### prepare shallow method #####
    criterion = get_loss(config['loss'], **config['loss_param'])
    criterion = criterion.to(device)

    ##### dataset preparation #####
    workers = 1 if config['dataset'] in ['gldv2delgembed'] else config['num_worker']
    train_loader, test_loader, db_loader = prepare_dataloader(config,
                                                              train_shuffle=False,
                                                              train_drop_last=False,
                                                              workers=workers)
    logging.info('Loading Training Data')
    train_data = torch.cat([x[0] for x in iter(train_loader)])
    logging.info(f'Number of Training Data: {len(train_data)}')
    # not delete train_loader, have to use for obtain codes and exporting train_out
    # del train_loader

    # train_loader = configs.dataloader(train_loader.dataset,
    #                                   config['batch_size'],
    #                                   shuffle=False,
    #                                   drop_last=False,
    #                                   workers=workers)

    logging.info('Begin Training')

    ##### shallow training #####
    criterion.train()
    train_codes, quan_error = criterion(train_data)
    # train_labels = train_data[1]

    logging.info(f'Quantization Error: {quan_error:.4f}')

    ##### obtain codes #####
    criterion.eval()
    criterion = criterion.to(device)
    test_out = obtain_codes(criterion, test_loader, device, return_id=calculate_mAP_using_id)
    db_out = obtain_codes(criterion, db_loader, device, return_id=calculate_mAP_using_id)
    train_out = obtain_codes(criterion, train_loader, device)

    if calculate_mAP_using_id:
        ground_truth_path = os.path.join(test_loader.dataset.root, 'ground_truth.csv')
        ground_truth = pd.read_csv(ground_truth_path)  # id = index id, images = images id in database

    ##### evaluation ######
    mAP = calculate_mAP(db_out['codes'], db_out['labels'],
                        test_out['codes'], test_out['labels'],
                        config['R'], device=device, onehot=onehot,
                        using_id=calculate_mAP_using_id,
                        ground_truth=ground_truth if calculate_mAP_using_id else None,
                        db_id=db_out['id'] if calculate_mAP_using_id else None,
                        test_id=test_out['id'] if calculate_mAP_using_id else None,
                        shuffle_database=config['shuffle_database'],
                        distance_func=config['distance_func'],
                        zero_mean=config['zero_mean_eval'])

    logging.info(f'mAP: {mAP:.6f}')

    ##### saving #####

    logdir = config['logdir']
    io.fast_save(db_out, f'{logdir}/outputs/db_out.pth')
    io.fast_save(test_out, f'{logdir}/outputs/test_out.pth')
    io.fast_save(train_out, f'{logdir}/outputs/train_out.pth')
    if config['save_model']:
        logging.info('Saving model')
        io.fast_save(criterion, f'{logdir}/models/last.pth')
    del db_out, test_out

    return mAP
