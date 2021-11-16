import numpy as np
import torch

from functions.hashing import get_distance_func
from functions.ternarization import tnt


def compute_distance(a, b, distance_mode='cosine'):
    return get_distance_func(distance_mode)(a, b)


def get_dist_stat(codes,
                  labels,
                  bs=1000,
                  ternary=False,
                  binary=True,
                  get_center=True,
                  get_hist=False,
                  hist_bin_size=64,
                  code_balance_stat=False,
                  quan_stat=False,
                  distance_mode='hamming',
                  minibatch_intraclass=False):
    if distance_mode != 'hamming':
        assert not get_hist, 'get_hist=True only for distance_mode=hamming'

    intracenter_avg = 0
    intracenter_std = 0
    intercenter_avg = 0
    intercenter_std = 0

    intraclass_avg = 0
    intraclass_std = 0
    interclass_avg = 0
    interclass_std = 0

    intra_count = 0
    intra_cent_count = 0

    inter_count = 0
    inter_cent_count = 0

    nbit = codes.size(1)
    nclass = labels.size(1)

    code_balance_avg = 0
    code_balance_std = 0
    quan_error_cs_avg = 0
    quan_error_cs_std = 0
    quan_error_l2_avg = 0
    quan_error_l2_std = 0

    intra_freq = torch.zeros(hist_bin_size + 1).to(codes.device)  # +1 to include last number, e.g. [0, ..., 64]
    inter_freq = torch.zeros(hist_bin_size + 1).to(codes.device)

    if code_balance_stat:
        code_balance = (codes.sign() == 1).float().mean(0)
        code_balance_avg = code_balance.mean().item()
        code_balance_std = code_balance.std().item()

    if quan_stat:
        quan_error_cs = torch.cosine_similarity(codes, codes.sign(), dim=1)  # .mean()
        quan_error_cs_avg = quan_error_cs.mean()
        quan_error_cs_std = quan_error_cs.std()

        quan_error_l2 = torch.norm(codes - codes.sign(), p=2, dim=1)  # .mean()
        quan_error_l2_avg = quan_error_l2.mean()
        quan_error_l2_std = quan_error_l2.std()

    if ternary:
        codes = tnt(codes.clone())
    elif binary:
        codes = codes.sign()

    center_codes = torch.zeros(nclass, nbit).to(codes.device)

    for c in range(nclass):
        print(f'processing class {c}', end='\r')
        # intramask = (labels.argmax(dim=1) == c)
        intramask = (labels[:, c] == 1)
        intracodes = codes[intramask]  # .sign()

        if ternary:
            center_codes[c] = tnt(intracodes.mean(dim=0, keepdim=True)).view(nbit)
        elif binary:
            center_codes[c] = intracodes.mean(dim=0).sign()
        else:
            center_codes[c] = intracodes.mean(dim=0)

        # intermask = (labels.argmax(dim=1) != c)
        intermask = ~intramask
        intercodes = codes[intermask]  # .sign()

        # intradist should be enough memory
        triu_mask = torch.ones(intracodes.size(0), intracodes.size(0)).bool()
        triu_mask = torch.triu(triu_mask, 1).to(intracodes.device)
        # intradist = (0.5 * (nbit - intracodes @ intracodes.t())) * triu_mask

        if not minibatch_intraclass:
            intradist = compute_distance(intracodes, intracodes, distance_mode) * triu_mask

            if get_hist:
                h = torch.histc(intradist[triu_mask.bool()], hist_bin_size + 1, 0, nbit)
                intra_freq += h

            # intradist = intradist.sum() / triu_mask.sum()
            intraclass_avg += intradist.sum().item()
            intraclass_std += (intradist ** 2).sum().item()
            intra_count += triu_mask.sum().item()

        else:
            intradist = 0
            intradist_std = 0
            triu_mask_numel = 0

            for bidx in range(intracodes.size(0) // bs + 1):
                print(f'processing class {c} [{bidx}/{intracodes.size(0) // bs + 1}]', end='\r')
                currbidx = bidx * bs
                nextbidx = (bidx + 1) * bs
                nextbidx = min(nextbidx, intracodes.size(0))

                if currbidx >= intracodes.size(0):  # already out of index
                    break

                batch_intra = intracodes[currbidx:nextbidx]
                intradist_ = compute_distance(batch_intra, intracodes, distance_mode) * triu_mask[currbidx:nextbidx].float()
                intradist += intradist_.sum()
                intradist_std += (intradist_**2).sum()
                triu_mask_numel += triu_mask[currbidx:nextbidx].sum()

                if get_hist:
                    h = torch.histc(intradist_[triu_mask[currbidx:nextbidx].bool()], hist_bin_size + 1, 0, nbit)
                    intra_freq += h

            intradist = intradist  # / triu_mask_numel
            intraclass_avg += intradist.item()
            intraclass_std += intradist_std.item()
            intra_count += triu_mask_numel.item()

        # dist_to_cent = (0.5 * (nbit - center_codes[c].view(1, -1) @ intracodes.t()))  # (1, N)
        dist_to_cent = compute_distance(center_codes[c].view(1, -1), intracodes, distance_mode)  # (1, N)
        intracenter_avg += dist_to_cent.view(-1).sum().item()
        intracenter_std += (dist_to_cent ** 2).view(-1).sum().item()
        intra_cent_count += intracodes.size(0)

        for bidx in range(intercodes.size(0) // bs + 1):
            print(f'processing class {c} [{bidx}/{intercodes.size(0) // bs + 1}]', end='\r')
            currbidx = bidx * bs
            nextbidx = (bidx + 1) * bs
            nextbidx = min(nextbidx, intercodes.size(0))

            if currbidx >= intercodes.size(0):  # already out of index
                break

            batch_inter = intercodes[currbidx:nextbidx]

            # interdist = (0.5 * (nbit - intracodes @ batch_inter.t()))
            interdist = compute_distance(intracodes, batch_inter, distance_mode)

            if get_hist:
                h = torch.histc(interdist, hist_bin_size + 1, 0, nbit)
                inter_freq += h

            inter_count += torch.numel(interdist)
            interclass_avg += interdist.sum().item()
            interclass_std += (interdist ** 2).sum().item()

            # dist_to_cent = (0.5 * (nbit - center_codes[c].view(1, -1) @ batch_inter.t()))  # (1, Nb)
            dist_to_cent = compute_distance(center_codes[c].view(1, -1), batch_inter, distance_mode)  # (1, Nb)
            intercenter_avg += dist_to_cent.view(-1).sum().item()
            intercenter_std += (dist_to_cent ** 2).view(-1).sum().item()
            inter_cent_count += batch_inter.size(0)

    interclass_avg /= inter_count
    interclass_std /= inter_count
    interclass_std -= (interclass_avg ** 2)

    intercenter_avg /= inter_cent_count
    intercenter_std /= inter_cent_count
    intercenter_std -= (intercenter_avg ** 2)

    intraclass_avg /= intra_count
    intraclass_std /= intra_count
    intraclass_std -= (intraclass_avg ** 2)

    intracenter_avg /= intra_cent_count
    intracenter_std /= intra_cent_count
    intracenter_std -= (intracenter_avg ** 2)

    print()
    ret = {
        'intraclass_avg': intraclass_avg,
        'intraclass_std': intraclass_std,
        'interclass_avg': interclass_avg,
        'interclass_std': interclass_std,
        'intracenter_avg': intracenter_avg,
        'intracenter_std': intracenter_std,
        'intercenter_avg': intercenter_avg,
        'intercenter_std': intercenter_std
    }

    if get_center:
        ret['center_codes'] = center_codes

    if code_balance_stat:
        ret['code_balance_avg'] = code_balance_avg
        ret['code_balance_std'] = code_balance_std

    if quan_stat:
        ret['quan_error_cs'] = quan_error_cs_avg
        ret['quan_error_cs_std'] = quan_error_cs_std
        ret['quan_error_l2'] = quan_error_l2_avg
        ret['quan_error_l2_std'] = quan_error_l2_std

    if get_hist:
        ret['intra_freq'] = intra_freq
        ret['inter_freq'] = inter_freq

    return ret


def get_dist_stat_embed(codes,
                        ids,
                        ground_truth,
                        index,
                        bs=1000,
                        ternary=False,
                        binary=True,
                        get_center=True,
                        get_hist=False,
                        hist_bin_size=64,
                        code_balance_stat=False,
                        quan_stat=False,
                        distance_mode='hamming'):
    if distance_mode != 'hamming':
        assert not get_hist, 'get_hist=True only for distance_mode=hamming'

    intracenter_avg = 0
    intercenter_avg = 0

    intraclass_avg = 0
    interclass_avg = 0
    intra_count = 0
    inter_count = 0
    inter_cent_count = 0

    nbit = codes.size(1)
    nquery = len(ground_truth)

    code_balance_avg = 0
    code_balance_std = 0
    quan_error_cs_avg = 0
    quan_error_cs_std = 0
    quan_error_l2_avg = 0
    quan_error_l2_std = 0

    intra_freq = torch.zeros(hist_bin_size + 1).to(codes.device)  # +1 to include last number, e.g. [0, ..., 64]
    inter_freq = torch.zeros(hist_bin_size + 1).to(codes.device)

    if code_balance_stat:
        code_balance = (codes.sign() == 1).float().mean(0)
        code_balance_avg = code_balance.mean().item()
        code_balance_std = code_balance.std().item()

    if quan_stat:
        quan_error_cs = torch.cosine_similarity(codes, codes.sign(), dim=1)  # .mean()
        quan_error_cs_avg = quan_error_cs.mean()
        quan_error_cs_std = quan_error_cs.std()

        quan_error_l2 = torch.norm(codes - codes.sign(), p=2, dim=1)  # .mean()
        quan_error_l2_avg = quan_error_l2.mean()
        quan_error_l2_std = quan_error_l2.std()

    if ternary:
        codes = tnt(codes.clone())
    elif binary:
        codes = codes.sign()

    center_codes = torch.zeros(nquery, nbit).to(codes.device)

    for c in range(nquery):
        print(f'processing class {c}', end='\r')
        intramask = np.isin(ids, ground_truth.iloc[c]['images'].split())
        intracodes = codes[intramask]  # .sign()

        if ternary:
            center_codes[c] = tnt(intracodes.mean(dim=0, keepdim=True)).view(nbit)
        elif binary:
            center_codes[c] = intracodes.mean(dim=0).sign()
        else:
            center_codes[c] = intracodes.mean(dim=0)

        # intermask = (labels.argmax(dim=1) != c)
        intermask = ~intramask[index]
        intercodes = codes[index][intermask]  # .sign()

        # intradist should be enough memory
        triu_mask = torch.ones(intracodes.size(0), intracodes.size(0))
        triu_mask = torch.triu(triu_mask, 1).to(intracodes.device)
        # intradist = (0.5 * (nbit - intracodes @ intracodes.t())) * triu_mask
        intradist = compute_distance(intracodes, intracodes, distance_mode) * triu_mask

        if get_hist:
            h = torch.histc(intradist[triu_mask.bool()], hist_bin_size + 1, 0, nbit)
            intra_freq += h

        intradist = intradist.sum() / triu_mask.sum()
        if triu_mask.sum() != 0:  # skip when only one code in intracodes, where no distance can be calculated
            intraclass_avg += intradist.item()
            intra_count += 1

        # dist_to_cent = (0.5 * (nbit - center_codes[c].view(1, -1) @ intracodes.t()))  # (1, N)
        dist_to_cent = compute_distance(center_codes[c].view(1, -1), intracodes, distance_mode)  # (1, N)
        intracenter_avg += dist_to_cent.mean().item()

        for bidx in range(intercodes.size(0) // bs + 1):
            print(f'processing class {c} [{bidx}/{intercodes.size(0) // bs + 1}]', end='\r')
            currbidx = bidx * bs
            nextbidx = (bidx + 1) * bs
            nextbidx = min(nextbidx, intercodes.size(0))

            if currbidx >= intercodes.size(0):  # already out of index
                break

            batch_inter = intercodes[currbidx:nextbidx]

            # interdist = (0.5 * (nbit - intracodes @ batch_inter.t()))
            interdist = compute_distance(intracodes, batch_inter, distance_mode)

            if get_hist:
                h = torch.histc(interdist, hist_bin_size + 1, 0, nbit)
                inter_freq += h

            inter_count += torch.numel(interdist)
            interclass_avg += interdist.sum().item()

            # dist_to_cent = (0.5 * (nbit - center_codes[c].view(1, -1) @ batch_inter.t()))  # (1, Nb)
            dist_to_cent = compute_distance(center_codes[c].view(1, -1), batch_inter, distance_mode)  # (1, Nb)
            intercenter_avg += dist_to_cent.sum().item()
            inter_cent_count += batch_inter.size(0)

    interclass_avg /= inter_count
    intercenter_avg /= inter_cent_count
    intraclass_avg /= intra_count
    intracenter_avg /= intra_count
    print()
    ret = {
        'intraclass_avg': intraclass_avg,
        'interclass_avg': interclass_avg,
        'intracenter_avg': intracenter_avg,
        'intercenter_avg': intercenter_avg,
    }

    if get_center:
        ret['center_codes'] = center_codes

    if code_balance_stat:
        ret['code_balance_avg'] = code_balance_avg
        ret['code_balance_std'] = code_balance_std

    if quan_stat:
        ret['quan_error_cs'] = quan_error_cs_avg
        ret['quan_error_cs_std'] = quan_error_cs_std
        ret['quan_error_l2'] = quan_error_l2_avg
        ret['quan_error_l2_std'] = quan_error_l2_std

    if get_hist:
        ret['intra_freq'] = intra_freq
        ret['inter_freq'] = inter_freq

    return ret
