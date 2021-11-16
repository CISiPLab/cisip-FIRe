import torch


def get_output_and_loss_adv(model, criterion, data, labels, index, onehot, loss_name, loss_cfg, stage, no_loss):
    x, code_logits, xrec, discs = model(data)
    if no_loss:
        loss = torch.tensor(0.)
    else:
        loss = criterion(x, code_logits, xrec, discs)

    return {'x': x, 'code_logits': code_logits}, loss


def update_meters_adv(model, meters, out, labels, onehot, criterion, loss_name, loss_cfg):
    acc = 0.0
    for key in criterion.losses:
        val = criterion.losses[key]
        if hasattr(val, 'item'):
            val = val.item()
        meters[key].update(val)
    meters['acc'].update(acc)


def optimize_params_adv(output, loss, optimizer,
                        method, model, criterion,
                        data, labels, index,
                        onehot, stage, loss_name, loss_cfg):
    if loss_name == 'tbh':
        opt_A, opt_C = optimizer

        actor_loss = criterion.losses['actor']
        critic_loss = criterion.losses['critic']

        params = [p for param in opt_A.param_groups for p in param['params']]
        actor_loss.backward(retain_graph=True, inputs=params)
        opt_A.step()  # step for main hashing flow

        params = [p for param in opt_C.param_groups for p in param['params']]
        critic_loss.backward(inputs=params)
        opt_C.step()  # step for discriminator
    else:
        raise NotImplementedError(f'No implementation for {loss_name} during adversarial optimization')
