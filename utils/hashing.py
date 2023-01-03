import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import engine
from utils.misc import Timer, compute_memory


def get_hamm_dist(codes, centroids, margin=0, normalize=False):
    with torch.no_grad():
        nbit = centroids.size(1)
        dist = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

        if normalize:
            dist = dist / nbit

        if margin == 0:
            return dist
        else:
            codes_clone = codes.clone()
            codes_clone[codes_clone.abs() < margin] = 0
            dist_margin = 0.5 * (nbit - torch.matmul(codes_clone.sign(), centroids.sign().t()))
            if normalize:
                dist_margin = dist_margin / nbit
            return dist, dist_margin


def get_codes_and_labels(model, loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vs = []
    ts = []
    for e, (d, t) in enumerate(loader):
        print(f'[{e + 1}/{len(loader)}]', end='\r')
        with torch.no_grad():
            # model forward
            d, t = d.to(device), t.to(device)
            v = model(d)
            if isinstance(v, tuple):
                v = v[0]

            vs.append(v)
            ts.append(t)

    print()
    vs = torch.cat(vs)
    ts = torch.cat(ts)
    return vs, ts


def hamming(a, b):
    nbit = a.size(1)
    return 0.5 * (nbit - torch.matmul(a, b.t()))  # (Na, nbit) * (nbit, Nb)


def euclidean(a, b):
    # dist = (a.unsqueeze(1) - b.unsqueeze(0)) ** 2
    # dist = dist.sum(dim=-1)
    # dist = dist ** 0.5
    # return dist
    return torch.cdist(a, b, p=2)  # (Na, Nb)


def cosine(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    return 1 - torch.matmul(a, b.t())


def get_distance_func(distance_func):
    if distance_func == 'hamming':
        return hamming
    elif distance_func == 'euclidean':
        return euclidean
    elif distance_func == 'cosine':
        return cosine
    else:
        raise ValueError(f'Distance function `{distance_func}` not implemented.')


def compute_distance(dist_f, gallery_codes, query_codes, bs=64, qbs=1000):
    timer = Timer()
    timer.tick()

    # dist = torch.zeros(len(query_codes), 0)
    dist = []
    with torch.no_grad():
        query_codes_ttd = engine.tensor_to_dataset(query_codes)
        query_loader = engine.dataloader(query_codes_ttd, qbs, False, 0, False)

        gallery_codes_ttd = engine.tensor_to_dataset(gallery_codes)
        gallery_loader = engine.dataloader(gallery_codes_ttd, bs, False, 0, False)

        # calculate hamming distance
        pbar = tqdm(enumerate(gallery_loader),
                    total=len(gallery_loader),
                    desc='Gallery Batch',
                    ascii=True,
                    bar_format='{l_bar}{bar:10}{r_bar}',
                    leave=False)

        for i, g_code in pbar:
            # dist = torch.cat((dist, dist_f(query_codes, g_code).cpu()), dim=1)
            qdist = []

            for j, q_code in enumerate(query_loader):
                qdist.append(dist_f(q_code, g_code).cpu())

            qdist = torch.cat(qdist, 0)

            dist.append(qdist)
            timer.toc()
            # print(f'Distance [{i + 1}/{len(gallery_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)  # .numpy()
        # print()
    return dist


def compute_distance_batch(dist_f, gallery_codes, query_codes, bs_g=32, bs_q=1024):
    timer = Timer()
    timer.tick()

    # dist = torch.zeros(len(query_codes), 0)
    dist = []
    with torch.no_grad():
        gallery_codes_ttd = engine.tensor_to_dataset(gallery_codes)
        query_codes_ttd = engine.tensor_to_dataset(query_codes)

        query_loader = engine.dataloader(query_codes_ttd, bs_q, False, 0, False)
        gallery_loader = engine.dataloader(gallery_codes_ttd, bs_g, False, 0, False)

        for j, q_code in enumerate(query_loader):
            # calculate hamming distance
            qdist = []
            for i, g_code in enumerate(gallery_loader):
                # dist = torch.cat((dist, dist_f(query_codes, g_code).cpu()), dim=1)
                qdist.append(dist_f(q_code, g_code).cpu())
                timer.toc()
                print(f'Distance [{j + 1}/{len(query_loader)}][{i + 1}/{len(gallery_loader)}] '
                      f'({timer.total:.2f}s)', end='\r')
            qdist = torch.cat(qdist, 1)
            dist.append(qdist)

        dist = torch.cat(dist, 0)
        print()
    return dist


def preprocess_on_codes(codes, threshold=0., sign=True, avoid_clone=False):
    # clone in case changing value of the original codes
    if not avoid_clone:
        codes = codes.clone()

    # if value within margin, set to 0, for ternary/dbq
    if threshold != 0:
        codes[codes.abs() < threshold] = 0

    if sign:
        codes = torch.sign(codes)  # (ndb, nbit)

    return codes


def get_rank(dist, R):
    # torch sorting is quite fast, pytorch ftw!!!
    topk_ids = torch.topk(dist, R, dim=1, largest=False, sorted=True)[1].cpu()
    return topk_ids


def calculate_mAP(db_codes, db_labels,
                  test_codes, test_labels,
                  Rs, threshold=0., dist_metric='hamming',
                  PRs=None,
                  landmark_gt=None, db_id=None, test_id=None,
                  multiclass=False):
    if landmark_gt is not None:
        assert db_id is not None and test_id is not None

    gc.collect()
    torch.cuda.empty_cache()

    avoid_clone = landmark_gt is not None

    ##### preprocess on codes #####
    db_codes = preprocess_on_codes(db_codes, threshold, dist_metric == 'hamming', avoid_clone)
    test_codes = preprocess_on_codes(test_codes, threshold, dist_metric == 'hamming', avoid_clone)
    db_labels = db_labels.cpu()  # .numpy()
    test_labels = test_labels.cpu()  # .numpy()

    if not multiclass:
        db_labels = db_labels.argmax(1)
        test_labels = test_labels.argmax(1)

    timer = Timer()
    total_timer = Timer()
    total_timer.tick()

    dist_f = get_distance_func(dist_metric)

    test_size = test_codes.size(0)
    db_size = db_codes.size(0)
    gb_size = 1024 * 1024 * 1024
    if test_size * db_size >= 1500000000:  # food101 is 75000 * 25000 ~= 1875M
        max_mem = 1 * gb_size  # approx using 16GB ram
    else:
        max_mem = 2 * gb_size
    tbs = min(int(max_mem / (db_size * 4)), test_size)
    mem_size = compute_memory(tbs, db_size, mode='GB')
    logging.info(f'Max approx. memory to store distance: {mem_size:.1f}GB, TBS: {tbs}')

    test_codes_ttd = engine.tensor_to_dataset(test_codes)
    test_loader = engine.dataloader(test_codes_ttd, tbs, False, 0, False)

    if isinstance(Rs, int):
        Rs = [Rs]
    elif not isinstance(Rs, list):
        try:
            Rs = [int(Rs)]
        except:
            Rs = [int(R) for R in Rs.split(',')]

    Rs = [R if R != -1 else len(db_codes) for R in Rs]
    maxR = max(Rs)
    APx = defaultdict(list)
    recalls = defaultdict(list)
    precisions = defaultdict(list)

    qbbar = tqdm(enumerate(test_loader),
                 total=len(test_loader),
                 desc='Query Batch',
                 ascii=True,
                 bar_format='{l_bar}{bar:10}{r_bar}')

    for ti, test_codes_batch in qbbar:
        ##### compute distance #####
        dist = compute_distance(dist_f, db_codes, test_codes_batch, qbs=tbs)

        ##### start sorting #####
        # fast sort
        # different sorting will have affect on mAP score! because the order with same hamming distance might be diff.
        # unsorted_ids = np.argpartition(dist, R - 1)[:, :R]

        # just sort by len(db_codes)
        # timer.tick()
        topk_ids = get_rank(dist, len(db_codes))
        # timer.toc()
        # logging.info(f'DB length={len(db_codes)}; Sorting ({timer.total:.2f}s)')

        ##### start mAP calculation #####
        timer.tick()
        if not multiclass and landmark_gt is None:
            ##### batch version #####
            batch_labels = test_labels[ti * tbs: (ti + 1) * tbs]  # (N, )
            batch_N = batch_labels.size(0)
            batch_idxs = topk_ids[:, :maxR].reshape(-1)  # (N * maxR)
            retrieved_labels = db_labels[batch_idxs].reshape(batch_N, maxR)  # (N, maxR)
            batch_imatch = torch.eq(retrieved_labels, batch_labels.unsqueeze(1))  # (N, maxR)
            batch_imatch_sum = torch.sum(batch_imatch, 1)  # (N, )
            batch_Lx = torch.cumsum(batch_imatch, 1)  # (N, maxR)
            batch_Px = batch_Lx.float() / torch.arange(1, maxR + 1, 1).unsqueeze(0)  # (N, maxR)

            for R in Rs:
                batch_rel = torch.sum(batch_imatch[:, :R], 1)  # (N, )
                batch_rel_mask = batch_imatch_sum != 0

                batch_ranking = batch_Px[:, :R] * batch_imatch[:, :R]  # (N, R)
                batch_AP = batch_ranking.sum(dim=1) / batch_rel.clamp(min=1)  # (if no relevant, clamp as 1)
                batch_AP = batch_AP * batch_rel_mask  # (set non relevant as 0)
                APx[R].extend(batch_AP.tolist())

            if PRs is not None:
                batch_Lx_for_recall = batch_Lx >= 1

                for Ri, R in enumerate(PRs):
                    batch_rel = torch.sum(batch_imatch[:, :R], 1)  # (N, )
                    recalls[R].extend(batch_Lx_for_recall[:, R - 1].tolist())
                    precisions[R].extend((batch_rel / R).tolist())
            # timer.toc()
            # todo: batch version for gldv2
        else:
            pbar = tqdm(range(dist.shape[0]), desc='Query', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
            ##### sequential version #####
            for i in pbar:
                if landmark_gt is not None:
                    test_img_id = test_id[i + ti * tbs]
                    already_predicted = set()
                    all_id_should_retrieve = set(
                        landmark_gt[landmark_gt['id'] == test_img_id]['images'].item().split(' '))

                    idx = topk_ids[i, :]
                    imatch = np.array([])
                    for img_id in db_id[idx[:maxR]]:
                        correct = img_id in all_id_should_retrieve and img_id not in already_predicted
                        imatch = np.append(imatch, correct)
                        already_predicted.add(img_id)
                    imatch_sum = np.sum(imatch)
                else:
                    label = test_labels[i + ti * tbs, :] * 2. - 1.
                    # label[label == 0] = -1
                    idx = topk_ids[i, :]
                    # idx = idx[np.argsort(dist[i, :][idx])]
                    imatch = np.sum(np.equal(db_labels[idx[:maxR], :].numpy(), label.numpy()), 1) > 0
                    imatch_sum = np.sum(imatch)

                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, maxR + 1, 1)

                for R in Rs:
                    rel = np.sum(imatch[:R])

                    if landmark_gt is None and imatch_sum == 0:  # no similar label at all, skip it?
                        continue

                    if landmark_gt is not None:
                        APx[R].append(np.sum(Px[:R] * imatch[:R]) / len(all_id_should_retrieve))
                    else:
                        if rel != 0:
                            APx[R].append(np.sum(Px[:R] * imatch[:R]) / rel)
                        else:  # didn't retrieve anything relevant
                            APx[R].append(0)

                if PRs is not None:
                    Lx[Lx >= 1] = 1
                    for Ri, R in enumerate(PRs):
                        rel = np.sum(imatch[:R])
                        if landmark_gt is None and imatch_sum == 0:  # no similar label at all, skip it?
                            continue

                        # recalls[R].append(float(rel > 0))
                        recalls[R].append(Lx[R - 1])
                        precisions[R].append(rel / R)

                timer.toc()

    # print()

    if PRs is not None:
        recalls = [np.mean(np.array(recalls[R])) for R in PRs]
        precisions = [np.mean(np.array(precisions[R])) for R in PRs]

    APx = {R: np.mean(np.array(APx[R])) for R in APx}
    mAPs = [APx[R] for R in Rs]
    if len(mAPs) == 1:
        mAPs = mAPs[0]

    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')
    ##### end mAP calculation #####

    ##### start recall and precision #####
    if PRs is not None:
        return mAPs, recalls, precisions

    return mAPs


def calculate_pr_curve(db_codes, db_labels,
                       test_codes, test_labels,
                       threshold=0., dist_metric='hamming'):
    gc.collect()
    torch.cuda.empty_cache()

    ##### preprocess on codes #####
    db_codes = preprocess_on_codes(db_codes, threshold, dist_metric == 'hamming')
    test_codes = preprocess_on_codes(test_codes, threshold, dist_metric == 'hamming')
    db_labels = db_labels.cpu()  # .numpy()
    test_labels = test_labels.cpu()  # .numpy()

    timer = Timer()
    total_timer = Timer()
    total_timer.tick()

    ##### compute distance #####
    dist_f = get_distance_func(dist_metric)
    dist = compute_distance(dist_f, db_codes, test_codes)

    ##### start sorting #####
    # just sort by len(db_codes)
    timer.tick()
    topk_ids = get_rank(dist, len(db_codes))
    timer.toc()
    logging.info(f'DB length={len(db_codes)}; Sorting ({timer.total:.2f}s)')
    ##### start sorting #####

    ##### start mAP calculation #####
    recalls = defaultdict(list)
    precisions = defaultdict(list)

    timer.tick()
    pbar = tqdm(range(dist.shape[0]), desc='Query', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')

    Rs = [1] + list(range(10, 101, 10)) + list(range(500, 10000, 500)) + list(range(10000, len(db_codes), 1000))

    test_db_labels = (torch.matmul(test_labels, db_labels.t()) > 0).bool()

    for i in pbar:
        label = test_labels[i, :]
        label[label == 0] = -1
        idx = topk_ids[i, :]
        # imatch = np.sum(np.equal(db_labels[idx, :], label), 1) > 0
        imatch = test_db_labels[i, idx].float()
        imatch_sum = torch.sum(imatch).item()
        imatch_cumsum = torch.cumsum(imatch, 0)

        for R in Rs:
            rel = imatch_cumsum[R - 1].item()

            precisions[R].append(rel / R)
            recalls[R].append(rel / imatch_sum)

        timer.toc()

    print()

    recalls = [np.mean(np.array(recalls[R])) for R in Rs]
    precisions = [np.mean(np.array(precisions[R])) for R in Rs]

    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')
    ##### end calculation #####

    return recalls, precisions, Rs


def sign_dist(inputs, centroids, margin=0):
    n, b1 = inputs.size()
    nclass, b2 = centroids.size()

    assert b1 == b2, 'inputs and centroids must have same number of bit'

    # sl = relu(margin - x*y)
    out = inputs.reshape(n, 1, b1) * centroids.sign().reshape(1, nclass, b1)
    out = torch.relu(margin - out)  # (n, nclass, nbit)

    return out


def calculate_similarity_matrix(centroids):
    nclass = centroids.size(0)
    sim = torch.zeros(nclass, nclass, device=centroids.device)

    for rc in range(nclass):
        for cc in range(nclass):
            sim[rc, cc] = (centroids[rc] == centroids[cc]).float().mean()

    return sim


def get_sim(label_a, label_b, onehot=True, soft_constraint=False):
    """
    label_a: (N, 1 or C)
    label_b: (M, 1 or C)

    return: boolean similarity (N, M)
    """
    if onehot:
        sim = torch.matmul(label_a.float(), label_b.float().t())
        S = sim >= 1
    else:
        n = label_a.size()
        m = label_b.size()

        label_a = label_a.reshape(n, 1)
        label_b = label_b.reshape(1, m)

        sim = label_a == label_b
        S = sim

    if soft_constraint:
        S = S.float()
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r

    return S


def log_trick(dot_product):
    """
    loss = log(1 + e^(dt)) - s * dt
    """
    return torch.log(1 + torch.exp(-torch.abs(dot_product))) + dot_product.clamp(min=0)
