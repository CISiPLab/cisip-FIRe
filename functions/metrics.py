import gc
import logging
import os
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

import configs
from functions.evaluate_roxf import DATASETS, configdataset, compute_map
from functions.hashing import inverse_sigmoid, get_distance_func
from functions.ternarization import tnt
from utils.misc import Timer


def calculate_mAP(db_codes, db_labels,
                  test_codes, test_labels,
                  Rs,
                  ternarization=None,
                  distance_func='hamming',
                  shuffle_database=False,
                  device=torch.device('cuda'),
                  onehot=True, using_id=False, ground_truth=None, db_id=None, test_id=None,
                  old_eval=False, avoid_clone=False, return_dist=False, zero_mean=False,
                  PRs: Sequence[int] = list()):
    if using_id:
        assert ground_truth is not None and db_id is not None and test_id is not None

    if not avoid_clone:
        # clone in case changing value of the original codes
        db_codes = db_codes.clone()
        test_codes = test_codes.clone()

    ##### start distance #####
    total_timer = Timer()
    total_timer.tick()

    logging.info('Start Preprocess')
    db_codes, test_codes = preprocess_for_calculate_mAP(db_codes, test_codes, ternarization, distance_func, zero_mean)

    logging.info('Start Distance')
    dist = compute_distances(db_codes, test_codes, distance_func, device)

    db_labels = db_labels.clone().cpu().numpy()
    test_labels = test_labels.clone().cpu().numpy()

    if Rs == -1:
        Rs = [len(db_codes)]
        logging.info(f'Computing mAP@All, R = {Rs[0]}')
    elif isinstance(Rs, int):
        Rs = [Rs]

    Rs = [len(db_codes) if x == -1 else x for x in Rs]  # make sure -1 in list also read as full
    logging.info(f'Computing mAP for R = {Rs}')

    if shuffle_database:
        logging.info('Shuffle Database Enabled.')
        randperm = torch.randperm(dist.size(1)).to(device)
        dist = dist[:, randperm]
        db_labels = db_labels[randperm.cpu().numpy()]
        # db_labels = db_labels[randperm]
        if using_id:
            db_id = db_id[randperm.cpu().numpy()]

    if dist.shape[0] * dist.shape[1] > 134217728:  # consider 4 bytes a tensor (32bit), for 512MB
        logging.info("Using CPU for dist, due to memory limitation")
        dist = dist.cpu()  # move to cpu first to avoid oom in gpu

    timer = Timer()

    mAPs, DistRs = [], []

    logging.info(f'Start Sorting')
    timer.tick()
    maxR = max(max(Rs), max(PRs) if PRs else 0)
    topk_ids = torch.topk(dist, maxR, dim=1, largest=False)[1].cpu()  # top k, k = largest R
    timer.toc()
    logging.info(f'Sort ({timer.total:.2f}s)')

    gc.collect()
    torch.cuda.empty_cache()

    # calculate mAP
    if using_id:
        output = compute_mAP_score_for_id_multi_R(dist, topk_ids,
                                                  db_id, test_id, ground_truth,
                                                  Rs, return_dist, PRs=PRs)
    else:
        output = compute_mAP_score_multi_R(dist, topk_ids,
                                           db_labels, test_labels, Rs, onehot, old_eval, return_dist, PRs=PRs)

    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')

    return output


def preprocess_for_calculate_mAP(db_codes, test_codes,
                                 ternarization=None, distance_func='hamming', zero_mean=False):
    ##### ternarize #####
    if ternarization is not None:
        # db_codes, test_codes = ternarize(db_codes, db_labels, test_codes, test_labels, **ternarization)[:2]
        mode = ternarization['mode']
        if distance_func == 'jmlh-dist':
            # we inverse jmlh sigmoid output back to normal input
            db_codes = inverse_sigmoid(db_codes)
            test_codes = inverse_sigmoid(test_codes)
            distance_func = 'hamming'  # we switch back to hamming distance because we are using normal input

        if mode == 'tnt':
            db_codes = tnt(db_codes)
            test_codes = tnt(test_codes)
        elif mode == 'threshold':
            threshold = ternarization['threshold']
            if threshold != 0:
                # if value within margin, set to 0
                db_codes[db_codes.abs() < threshold] = 0
                test_codes[test_codes.abs() < threshold] = 0

    ##### zero mean for code balance #####
    if zero_mean:
        logging.info('Zero mean enabled.')
        db_codes_mean = db_codes.mean(dim=0, keepdim=True)
        db_codes = db_codes - db_codes_mean
        test_codes = test_codes - db_codes_mean

    ##### binarize #####
    if distance_func == 'hamming':  # jhml output is {0, 1}, we can skip this step
        # binarized
        db_codes = torch.sign(db_codes)  # (ndb, nbit)
        test_codes = torch.sign(test_codes)  # (nq, nbit)

    return db_codes, test_codes


def compute_mAP_score_multi_R(dist,
                              topk_ids,
                              db_labels,
                              test_labels,
                              Rs,
                              onehot=True,
                              old_eval=False,
                              return_dist=False,
                              PRs: Sequence[int] = list()):
    Dists = []
    APx = defaultdict(list)
    recalls = defaultdict(list)
    precisions = defaultdict(list)
    maxR = max(max(Rs), max(PRs) if PRs else 0)
    pbar = tqdm(range(dist.size(0)), desc='Query', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)
    for i in pbar:
        if onehot:
            label = test_labels[i, :]  # [0,1,0,0] one hot label
            label[label == 0] = -1
            idx = topk_ids[i, :]
            # idx = idx[np.argsort(dist[i, :][idx])]
            # imatch = (db_labels[idx[:R]] @ label) > 0  # (R, C) dot (C, 1) -> (R,)
            imatch = np.sum(np.equal(db_labels[idx[:maxR], :], label), 1) > 0
        else:
            label = test_labels[i]
            idx = topk_ids[i, :]
            imatch = (db_labels[idx[0: maxR]] == label) > 0

        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, maxR + 1, 1)  # ap += num_correct / (i + 1)

        for R in Rs:
            rel = np.sum(imatch[:R])
            if rel != 0:
                APx[R].append(np.sum(Px[:R] * imatch[:R]) / rel)
            elif not old_eval:
                APx[R].append(0)

        if PRs:
            Lx[Lx > 1] = 1
            for Ri, R in enumerate(PRs):
                rel = np.sum(imatch[:R])

                recalls[R].append(Lx[R - 1])
                precisions[R].append(rel / R)

        if return_dist:
            Dists.append(dist[i, idx])

    if PRs:
        recalls = [np.mean(np.array(recalls[R])) for R in PRs]
        precisions = [np.mean(np.array(precisions[R])) for R in PRs]

    APx = {R: np.mean(np.array(APx[R])) for R in APx}
    mAPs = [APx[R] for R in Rs]
    if len(mAPs) == 1:
        mAPs = mAPs[0]

    if return_dist and PRs:
        return mAPs, recalls, precisions, [torch.stack(Dists).cpu().numpy()[:, :R] for R in Rs]
    elif return_dist and not PRs:
        return mAPs, [torch.stack(Dists).cpu().numpy()[:, :R] for R in Rs]
    elif PRs:
        return mAPs, recalls, precisions
    else:
        return mAPs


def compute_mAP_score_for_id_multi_R(dist,
                                     topk_ids,
                                     db_id,
                                     test_id,
                                     ground_truth,
                                     Rs,
                                     return_dist=False,
                                     PRs: Sequence[int] = list()):
    Dists = []
    APx = defaultdict(list)
    recalls = defaultdict(list)
    precisions = defaultdict(list)
    maxR = max(max(Rs), max(PRs) if PRs else 0)
    pbar = tqdm(range(dist.size(0)), desc='Query', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)
    for i in pbar:
        already_predicted = set()
        test_img_id = test_id[i]
        all_id_should_retrieve = set(ground_truth[ground_truth['id'] == test_img_id]['images'].item().split(" "))
        idx = topk_ids[i, :]
        imatch = np.array([])
        for img_id in db_id[idx[0: maxR]]:
            correct = img_id in all_id_should_retrieve and img_id not in already_predicted
            imatch = np.append(imatch, correct)
            already_predicted.add(img_id)
        # imatch = np.array([db_id in all_id_should_retrieve for db_id in db_id[idx[0: R]]])

        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, maxR + 1, 1)  # ap += num_correct / (i + 1)
        # Px = Lx.float() / torch.arange(1, R + 1, 1).to(device)

        # https://github.com/tensorflow/models/blob/7f0ee4cb1f10d4ada340cc5bfe2b99d0d690b219/research/delf/delf/python/datasets/google_landmarks_dataset/metrics.py#L160
        for R in Rs:
            rel = np.sum(imatch[:R])
            APx[R].append(np.sum(Px[:R] * imatch[:R]) / len(all_id_should_retrieve))

        if PRs:
            Lx[Lx > 1] = 1
            for Ri, R in enumerate(PRs):
                rel = np.sum(imatch[:R])

                recalls[R].append(Lx[R - 1])
                precisions[R].append(rel / R)

        if return_dist:
            Dists.append(dist[i, idx])

    if PRs:
        recalls = [np.mean(np.array(recalls[R])) for R in PRs]
        precisions = [np.mean(np.array(precisions[R])) for R in PRs]

    APx = {R: np.mean(np.array(APx[R])) for R in APx}
    mAPs = [APx[R] for R in Rs]
    if len(mAPs) == 1:
        mAPs = mAPs[0]

    if return_dist and PRs:
        return mAPs, recalls, precisions, [torch.stack(Dists).cpu().numpy()[:, :R] for R in Rs]
    elif return_dist and not PRs:
        return mAPs, [torch.stack(Dists).cpu().numpy()[:, :R] for R in Rs]
    elif PRs:
        return mAPs, recalls, precisions
    else:
        return mAPs


def compute_distances(db_codes, test_codes, distance_func, device):
    dist = []
    dist_f = get_distance_func(distance_func)
    with torch.no_grad():
        db_codes = db_codes.to(device)
        test_codes = test_codes.to(device)

        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        pbar = tqdm(db_codes_loader, desc='Distance', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                    disable=configs.disable_tqdm)
        for i, db_code in enumerate(pbar):
            dist.append(dist_f(test_codes, db_code).cpu())  # move to gpu avoid oom

        dist = torch.cat(dist, 1)  # .numpy()
    return dist


def calculate_mAP_roxf(db_codes, test_codes, test_dataset,
                       ternarization=None,
                       distance_func='hamming',
                       device=torch.device('cuda')):
    assert test_dataset in DATASETS
    # evaluate ranks
    ks = [1, 5, 10]
    # Set test dataset: roxford5k | rparis6k

    logging.info('>> {}: Evaluating test dataset...'.format(test_dataset))
    # config file for the dataset
    # separates query image list from database image list, when revisited protocol used
    cfg = configdataset(test_dataset, os.path.join('data'))

    # clone in case changing value of the original codes
    db_codes = db_codes.clone()
    test_codes = test_codes.clone()

    logging.info('Start Preprocess')
    if ternarization is not None:
        # db_codes, test_codes = ternarize(db_codes, db_labels, test_codes, test_labels, **ternarization)[:2]
        mode = ternarization['mode']
        if mode == 'tnt':
            db_codes = tnt(db_codes)
            test_codes = tnt(test_codes)
        elif mode == 'threshold':
            threshold = ternarization['threshold']
            if threshold != 0:
                # if value within margin, set to 0
                db_codes[db_codes.abs() < threshold] = 0
                test_codes[test_codes.abs() < threshold] = 0

    if distance_func == 'hamming':  # jhml output is {0, 1}, we can skip this step
        # binarized
        db_codes = torch.sign(db_codes)  # (ndb, nbit)
        test_codes = torch.sign(test_codes)  # (nq, nbit)

    dist_f = get_distance_func(distance_func)
    dist = []

    timer = Timer()
    total_timer = Timer()

    total_timer.tick()

    logging.info('Start Distance')

    with torch.no_grad():
        db_codes = db_codes.to(device)
        test_codes = test_codes.to(device)

        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        pbar = tqdm(db_codes_loader, desc='Distance', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                    disable=configs.disable_tqdm)
        for i, db_code in enumerate(pbar):
            dist.append(dist_f(test_codes, db_code).cpu())  # move to gpu avoid oom

        dist = torch.cat(dist, 1)  # .numpy()

    logging.info('Start Sorting')
    # fast sort
    timer.tick()
    if dist.shape[0] * dist.shape[1] > 134217728:  # consider 4 bytes a tensor (32bit), for 512MB
        logging.info("Using CPU for dist, due to memory limitation")
        dist = dist.cpu()  # move to cpu first to avoid oom in gpu
    # ranks = torch.topk(dist, min(max(ks)*1000, dist.shape[0]), dim=1, largest=False)[1].cpu()
    ranks = torch.argsort(dist, dim=1).t()
    timer.toc()
    logging.info(f'Sort ({timer.total:.2f}s)')

    # revisited evaluation
    gnd = cfg['gnd']

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['easy']]),
             'junk': np.concatenate([gnd[i]['junk'], gnd[i]['hard']])}
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['easy'], gnd[i]['hard']]),
             'junk': np.concatenate([gnd[i]['junk']])}
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {'ok': np.concatenate([gnd[i]['hard']]),
             'junk': np.concatenate([gnd[i]['junk'], gnd[i]['easy']])}
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    logging.info('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapE * 100, decimals=2),
                                                         np.around(mapM * 100, decimals=2),
                                                         np.around(mapH * 100, decimals=2)))
    logging.info(
        '>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprE * 100, decimals=2),
                                                   np.around(mprM * 100, decimals=2),
                                                   np.around(mprH * 100, decimals=2)))
    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')

    return mapE, mapM, mapH, apsE, apsM, apsH, mprE, mprM, mprH
