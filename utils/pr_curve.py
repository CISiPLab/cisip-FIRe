import numpy as np
from tqdm import tqdm

import configs
from functions.hashing import get_hamm_dist

draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]


def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    #  Detail usage: https://github.com/swuxyj/DeepHash-pytorch/blob/master/utils/precision_recall_curve.py 
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(get_hamm_dist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range, disable=configs.disable_tqdm):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R

