import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import configs
from functions.loss.itq import ITQLoss


def obtain_features(model, config, loader):
    train_dataset = configs.dataset(config,
                                    filename='train.txt',
                                    transform_mode='test',
                                    skip_preprocess=config['dataset_kwargs'].get('train_skip_preprocess', False))
    train_loader = configs.dataloader(train_dataset, config['batch_size'], shuffle=False, drop_last=False)

    model.eval()

    pbar = tqdm(train_loader, desc='Obtain Features', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)

    ret_codes = []

    device = next(model.parameters()).device

    for i, (data, labels, index) in enumerate(pbar):
        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)

            x, code_logits, rec_x = model(data)[:3]

            ret_codes.append(x.cpu())

    return torch.cat(ret_codes), device


class UHBDNNLoss(nn.Module):
    def __init__(self, quan, independence, balance, **kwargs):
        super().__init__()

        self.register_buffer('B', None)
        self.quan = quan
        self.independence = independence
        self.balance = balance
        self.nbit = kwargs['nbit']

        self.losses = {}

    def prepare_dataset_from_model(self, model, config, train_loader, test_loader, db_loader):
        train_features, device = obtain_features(model, config, train_loader)

        itq = ITQLoss(self.nbit, 50)
        itq.train()
        self.B = itq(train_features)[0]

    def forward(self, x, code_logits, rec_x, index):
        rec_loss = F.mse_loss(rec_x, x)

        quan_error = F.mse_loss(code_logits, self.B[index].to(code_logits.device))

        orthogonal = (code_logits.t() @ code_logits) / code_logits.size(0)
        independence_error = F.mse_loss(orthogonal, torch.eye(self.nbit, device=code_logits.device),
                                        reduction='sum')

        balance_error = code_logits.sum(dim=0).pow(2).mean()

        total_loss = (rec_loss +
                      self.quan * quan_error +
                      self.independence * independence_error +
                      self.balance * balance_error)

        self.losses['rec'] = rec_loss
        self.losses['quan'] = quan_error
        self.losses['ind'] = independence_error
        self.losses['bal'] = balance_error

        return total_loss
