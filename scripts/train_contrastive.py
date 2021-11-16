import torch


def get_output_and_loss_contrastive(model, criterion, data, labels, index, onehot, loss_name, loss_cfg, stage, no_loss):
    if model.training:
        imgi, imgj = data[:, 0], data[:, 1]
        prob_i, prob_j, z_i, z_j = model((imgi, imgj))
        loss = criterion(prob_i, prob_j, z_i, z_j)
        return {
                   # 'z_i': z_i,
                   # 'z_j': z_j
               }, loss
    else:
        imgi = data
        z = model(imgi)
        return {
                    'code_logits': z,
               }, torch.tensor(0.)


def update_meters_contrastive(model, meters, out, labels, onehot, criterion, loss_name, loss_cfg):
    acc = 0.0
    for key in criterion.losses:
        val = criterion.losses[key]
        if hasattr(val, 'item'):
            val = val.item()
        meters[key].update(val)
    meters['acc'].update(acc)
