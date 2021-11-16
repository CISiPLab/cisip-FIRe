import logging
import os
import numpy as np
import pickle

"""
Reference: 
https://github.com/filipradenovic/revisitop/blob/e1c9657fd5019bfc64c5fa1ac02cfd74b2f3368c/python/evaluate.py
"""


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


DATASETS = ['roxford5k', 'rparis6k', 'roxford5kdelgembed', 'rparis6kdelgembed']  # revisitop1m


def configdataset(dataset, dir_main):
    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    if dataset in ['roxford5k', 'rparis6k', 'roxford5kdelgembed', 'rparis6kdelgembed']:
        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'

    elif dataset == 'revisitop1m':
        # loading imlist from a .txt file
        cfg = {}
        cfg['imlist_fname'] = os.path.join(dir_main, dataset, '{}.txt'.format(dataset))
        cfg['imlist'] = read_imlist(cfg['imlist_fname'])
        cfg['qimlist'] = []
        cfg['ext'] = ''
        cfg['qext'] = ''

    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg


def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


def read_imlist(imlist_fn):
    with open(imlist_fn, 'r') as file:
        imlist = file.read().splitlines()
    return imlist


