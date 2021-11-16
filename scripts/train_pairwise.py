import logging

import torch


def update_params_pairwise(loss_param, train_loader, nbit, nclass):
    # update pairwise loss parameters
    loss_param['loss_param'].update({
        'keep_train_size': loss_param['loss_param'].get('keep_train_size', False),
        'train_size': len(train_loader.dataset),
        'nbit': nbit,
        'nclass': nclass
    })

    if loss_param['loss_param']['keep_train_size']:
        logging.info('Keep train size!')


def get_output_and_loss_pairwise(model, criterion, data, labels, index, onehot, loss_name, loss_cfg, stage='train',
                                 no_loss=False):
    logits, code_logits = model(data)
    if no_loss:
        loss = torch.tensor(0.)
    elif stage == 'train':
        if loss_cfg.get('train_size', 0) != 0 and loss_cfg.get('keep_train_size', False):
            ind = index
        else:
            ind = None
        loss = criterion(code_logits, labels, ind)

    elif stage == 'test':
        ind = None  # no need to put ind into criterion during testing
        if loss_name in ['dfh']:
            loss = torch.tensor(0.)
        else:
            loss = criterion(code_logits, labels, ind)

    else:
        raise ValueError('only train and test can be set as stage')
    return {
               'logits': logits,
               'code_logits': code_logits
           }, loss


def update_meters_pairwise(model, meters, out, labels, onehot, criterion, loss_name, loss_cfg):
    acc = 0.0
    for key in criterion.losses:
        val = criterion.losses[key]
        if hasattr(val, 'item'):
            val = val.item()
        meters[key].update(val)
    meters['acc'].update(acc)
