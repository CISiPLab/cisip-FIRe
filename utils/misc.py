import time

from PIL import Image
from torch.nn import DataParallel


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.sumsq = 0
        self.avgsumsq = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.sumsq += (val * n) ** 2
        self.avgsumsq = self.sumsq / self.count  # (E[X^2])
        self.std = (self.avgsumsq - self.avg ** 2 + 1e-8) ** 0.5  # sqrt(E[X^2] - E[X]^2)


class Timer(object):
    def __init__(self):
        self.start = 0
        self.end = 0
        self.total = 0

    def tick(self):
        self.start = time.time()
        return self.start

    def toc(self):
        self.end = time.time()
        self.total = self.end - self.start
        return self.end

    def print_time(self, title):
        print(f'{title} time: {self.total:.4f}s')


def to_list(v):
    if not isinstance(v, list):
        return [v]
    return v


class DataParallelPassthrough(DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, is_cpu=False):
        super(DataParallel, self).__init__()
        if is_cpu:
            self.module = module
            self.device_ids = []
            return
        else:
            super(DataParallelPassthrough, self).__init__(module, device_ids=device_ids,
                                                          output_device=output_device, dim=dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


class _dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dot_dict(d):
    outd = dict()

    for k, v in d.items():
        if isinstance(v, dict):
            outd[k] = dot_dict(v)
        else:
            outd[k] = v

    return _dot_dict(outd)


def pil_loader(f) -> Image.Image:
    img = Image.open(f)
    return img.convert('RGB')


if __name__ == '__main__':
    x = {
        'a': {
            'b': {
                'c': 123
            }
        },
        'b': 456
    }
    xd = dot_dict(x)
    print(xd)
    print(xd.a)
    print(xd.a.b)
    print(xd.a.b.c)
    print(xd.b)
    print(xd.c)