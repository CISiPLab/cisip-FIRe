import math

from torch.optim.lr_scheduler import LambdaLR


def no_decay(optimizer):
    return LambdaLR(optimizer,
                    lambda ep: 1)


def cosine_decay_linear_warmup(optimizer, epochs, warmup_epochs):
    def csw(epoch, epochs, warmup_epochs):
        epoch = epoch + 1
        if epoch <= warmup_epochs:
            scale = epoch / max(1., warmup_epochs)
        else:
            scale = 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / max(1., epochs - warmup_epochs + 1)))
        return max(0., scale)

    return LambdaLR(optimizer,
                    lambda epoch: csw(epoch, epochs, warmup_epochs))
