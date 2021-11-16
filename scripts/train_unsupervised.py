import torch


def get_output_and_loss_unsupervised(model,
                                     criterion,
                                     data,
                                     labels,
                                     index,
                                     onehot,
                                     loss_name,
                                     loss_cfg,
                                     stage,
                                     no_loss):
    x, code_logits, b = model(data)[:3]

    if not no_loss:
        if loss_name in ['ssdh']:  # test stage ssdh
            if stage == 'train':
                loss = criterion(x, code_logits, b, labels, index)
            else:
                loss = torch.tensor(0.)

        elif loss_name in ['uhbdnn']:
            loss = criterion(x, code_logits, b, index)

        else:
            try:
                loss = criterion(x, code_logits, b, labels, index)
            except:
                raise NotImplementedError(f'Loss name: {loss_name}; Stage: {stage}')
    else:
        loss = torch.tensor(0.)

    return {
               'x': x,
               'code_logits': code_logits,
               'b': b
           }, loss


def update_meters_unsupervised(model, meters, out, labels, onehot, criterion, loss_name, loss_cfg):
    # acc = 0.0
    for key in criterion.losses:
        val = criterion.losses[key]
        if hasattr(val, 'item'):
            val = val.item()
        meters[key].update(val)
    # meters['acc'].update(acc)
