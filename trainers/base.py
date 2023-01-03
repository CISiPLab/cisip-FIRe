import logging
from collections import defaultdict

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import engine
from utils import io
from utils.metrics import calculate_accuracy
from utils.misc import AverageMeter


class BaseTrainer:
    def __init__(self, config: DictConfig):
        self.config = config

        self.dataset = None
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.current_epoch = 0
        self.inference_datakey = ''

        self.device = torch.device(config['device'])

    def load_criterion(self):
        self.criterion = hydra.utils.instantiate(self.config.criterion)

    def finetune_setup(self, *args, **kwargs):
        """
        for fine-tuning after pre-training
        """
        pass

    def get_learning_rate(self):
        if self.scheduler is None:
            return [0]

        return self.scheduler.get_last_lr()

    def load_for_inference(self, logdir):
        """
        if trainer is initiated by non-training mode, then this is to load all neccessary stuff.
        basically doing prepare_before_training
        :param logdir:
        :return:
        """
        pass

    def to_device(self, device=None):
        if device is None:
            device = self.device

        if self.model is not None:
            self.model = self.model.to(device)

        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

    def is_ready_for_training(self):
        r = True

        for item in [self.dataset,
                     self.dataloader,
                     self.model,
                     self.optimizer,
                     self.scheduler,
                     self.criterion]:
            r = r and item is not None

        return r

    def is_ready_for_inference(self):
        r = True

        for item in [self.dataset,
                     self.dataloader,
                     self.model,
                     self.criterion]:
            r = r and item is not None

        return r

    def load_dataset(self, load_db=True):
        logging.info('Creating Datasets')
        train_dataset = hydra.utils.instantiate(self.config.dataset.train_dataset)
        test_dataset = hydra.utils.instantiate(self.config.dataset.test_dataset)
        db_dataset = hydra.utils.instantiate(self.config.dataset.db_dataset)

        logging.info(f'Number of Train data: {len(train_dataset)}')
        logging.info(f'Number of DB data: {len(db_dataset)}')
        logging.info(f'Number of Query data: {len(test_dataset)}')

        self.dataset = {
            'train': train_dataset,
            'test': test_dataset,
            'db': db_dataset
        }

    def load_dataloader(self):
        assert self.dataset is not None
        bs = self.config.batch_size
        train_loader = engine.dataloader(self.dataset['train'], bs, shuffle=True, drop_last=True)
        train_no_shuffle = engine.dataloader(self.dataset['train'], bs, shuffle=False, drop_last=False)
        test_loader = engine.dataloader(self.dataset['test'], self.config.batch_size, shuffle=False, drop_last=False)
        db_loader = engine.dataloader(self.dataset['db'], self.config.batch_size, shuffle=False, drop_last=False)

        self.dataloader = {
            'train': train_loader,
            'train_no_shuffle': train_no_shuffle,
            'test': test_loader,
            'db': db_loader
        }

    def load_model(self):
        logging.info('Creating Model')
        model = hydra.utils.instantiate(self.config.model)
        self.model = model

    def load_optimizer_and_scheduler(self):
        assert self.model is not None

        lr = self.config.optim.lr
        backbone_lr_scale = self.config.backbone_lr_scale

        params = [{'params': self.model.get_backbone().parameters(),
                   'lr': lr * backbone_lr_scale},
                  {'params': self.model.get_training_modules().parameters()}]

        if backbone_lr_scale == 0:  # if not training backbone, freeze it
            logging.info('Freezing backbone')
            backbone_params = params.pop(0)['params']
            for p in backbone_params:
                p.requires_grad_(False)

        self.optimizer = hydra.utils.instantiate(self.config.optim, params)
        self.scheduler = hydra.utils.instantiate(self.config.scheduler, self.optimizer)

    def prepare_before_training(self):
        pass

    def prepare_before_first_epoch(self):
        pass

    def save_before_training(self, logdir):
        with open(f'{logdir}/config.yaml', 'w+') as f:
            # f.write(OmegaConf.to_yaml(self.config))

            f.write(yaml.dump(OmegaConf.to_object(self.config)))

    def save_codes(self, codes, fn):
        io.fast_save(codes, fn)

    def load_codes(self, fn):
        return torch.load(fn)

    def save_model_state(self, fn):
        modelsd = self.model.state_dict()
        modelsd = {k: v.clone().cpu() for k, v in modelsd.items()}
        io.fast_save(modelsd, fn)

    def load_model_state(self, fn):
        modelsd = torch.load(fn, map_location='cpu')
        self.model.load_state_dict(modelsd)

    def save_training_state(self, fn):
        optimsd = self.optimizer.state_dict()
        schedulersd = self.scheduler.state_dict()
        io.fast_save({'optim': optimsd,
                      'scheduler': schedulersd}, fn)

    def load_training_state(self, fn):
        sd = torch.load(fn, map_location='cpu')
        self.optimizer.load_state_dict(sd['optim'])
        self.scheduler.load_state_dict(sd['scheduler'])

    def compute_features(self, datakey):
        self.model.eval()
        ret = defaultdict(list)

        with torch.no_grad():
            with tqdm(self.dataloader[datakey], bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
                for i, data in enumerate(tepoch):
                    _, output = self.compute_features_one_batch(data)
                    output['labels'] = data[1]  # labels

                    for key in output:
                        ret[key].append(output[key].cpu())

            res = {}
            for key in ret:
                if isinstance(ret[key][0], torch.Tensor):
                    res[key] = torch.cat(ret[key])
                else:
                    res[key] = np.concatenate(ret[key])
        return res

    def compute_features_one_batch(self, data):
        device = self.device

        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        data = image, labels, index
        output = self.model(image)

        return data, self.parse_model_output(output)

    def parse_model_output(self, output):
        logits, codes = output
        return {
            'logits': logits,
            'codes': codes,
        }

    def inference_one_batch(self, *args, **kwargs):
        data, meters = args

        with torch.no_grad():
            data, output = self.compute_features_one_batch(data)

            image, labels, index = data
            logits, codes = output['logits'], output['codes']

            loss = self.criterion(logits, codes, labels)
            acc = calculate_accuracy(logits, labels)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))
            meters['acc'].update(acc.item(), image.size(0))

        return {
            'codes': codes,
            'labels': labels
        }

    def inference_one_epoch(self, datakey='test', return_codes=False, **kwargs):
        assert self.is_ready_for_inference()

        self.model.eval()
        self.criterion.eval()
        meters = defaultdict(AverageMeter)

        ret = defaultdict(list)

        with tqdm(self.dataloader[datakey], bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            self.inference_datakey = datakey

            for i, data in enumerate(tepoch):
                output = self.inference_one_batch(data, meters, bidx=i, **kwargs)
                tepoch.set_postfix({k: v.avg for k, v in meters.items()})

                if return_codes:
                    for key in output:
                        ret[key].append(output[key].cpu())

        if return_codes:
            res = {}
            for key in ret:
                if isinstance(ret[key][0], torch.Tensor):
                    res[key] = torch.cat(ret[key])
                else:
                    res[key] = np.concatenate(ret[key])
            return meters, res

        return meters

    def train_one_batch(self, *args, **kwargs):
        """
        Args:
            args: [data, meters]
            kwargs: {'ep': current epoch, 'bidx': current batch index}
        """
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        logits, codes = self.model(image)

        loss = self.criterion(logits, codes, labels)

        # backward and update
        loss.backward()
        self.optimizer.step()

        acc = calculate_accuracy(logits, labels)

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
        meters['acc'].update(acc.item(), image.size(0))

    def train_one_epoch(self, **kwargs):
        """
        Args:
            kwargs: {'ep': current epoch}
        """
        assert self.is_ready_for_training()

        self.model.train()
        self.criterion.train()
        meters = defaultdict(AverageMeter)

        with tqdm(self.dataloader['train'], bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            for i, data in enumerate(tepoch):
                self.train_one_batch(data, meters, bidx=i, **kwargs)
                tepoch.set_postfix({k: v.avg for k, v in meters.items()})

        self.scheduler.step()

        return meters
