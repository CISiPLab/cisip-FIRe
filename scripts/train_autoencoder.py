import torch


def get_output_and_loss_autoencoder(model, criterion, data, labels, index, onehot, loss_name, loss_cfg, stage, no_loss):
    x, code_logits, xrec, code_rec, L = model(data)
    if no_loss:
        loss = torch.tensor(0.)
    else:
        loss = criterion(x, code_logits, xrec, code_rec, L)
    return {
               'x': x,
               'code_logits': code_logits,
               'xrec': xrec,
               'code_rec': code_rec,
               'L': L
           }, loss


def update_meters_autoencoder(model, meters, out, labels, onehot, criterion, loss_name, loss_cfg):
    acc = 0.0
    for key in criterion.losses:
        val = criterion.losses[key]
        if hasattr(val, 'item'):
            val = val.item()
        meters[key].update(val)
    meters['acc'].update(acc)
