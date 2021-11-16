import torch


def tnt(x):
    x_abs = torch.abs(x)
    x_abs_sorted, x_sorted_index = torch.sort(x_abs, dim=-1, descending=True)

    # unit norm
    x_abs_sorted_norm = x_abs_sorted / (x_abs_sorted.norm(p=2, dim=-1, keepdim=True) + 1e-7)

    # perform cosine similarity search
    cs = torch.cumsum(x_abs_sorted_norm, dim=-1)  # cumsum last dim
    cs = cs / torch.sqrt(torch.arange(1, cs.size(-1) + 1).to(cs.device).float())

    # search for max cosine similarity
    max_cs, max_index = torch.max(cs, dim=-1, keepdim=True)
    x_thresh = torch.gather(x_abs_sorted, dim=-1, index=max_index)

    # ternarize
    t = torch.sign(x)
    t[x_abs < x_thresh] = 0
    return t
