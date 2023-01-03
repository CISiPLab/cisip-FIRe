import hydra
import torch

from trainers.base import BaseTrainer


class TBHTrainer(BaseTrainer):
    def __init__(self, config):
        super(TBHTrainer, self).__init__(config)

        self.adv_optimizer = None
        self.adv_scheduler = None

    def load_optimizer_and_scheduler(self):
        super().load_optimizer_and_scheduler()

        param_groups = [{'params': self.model.get_discriminator_modules().parameters()}]
        self.adv_optimizer = hydra.utils.instantiate(self.config.optim, param_groups)
        self.adv_scheduler = hydra.utils.instantiate(self.config.scheduler, self.adv_optimizer)

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            feats, codes, rec_feats, discs = self.model(image)
            loss = self.criterion(feats, codes, rec_feats, discs)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

        return {
            'codes': codes - 0.5,  # tbh output is sigmoid
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()
        self.adv_optimizer.zero_grad()

        feats, codes, rec_feats, discs = self.model(image)
        loss = self.criterion(feats, codes, rec_feats, discs)

        actor_loss = self.criterion.losses['actor']
        critic_loss = self.criterion.losses['critic']

        params = [p for param in self.optimizer.param_groups for p in param['params']]
        actor_loss.backward(retain_graph=True, inputs=params)
        self.optimizer.step()  # step for main hashing flow

        params = [p for param in self.adv_optimizer.param_groups for p in param['params']]
        critic_loss.backward(inputs=params)
        self.adv_optimizer.step()  # step for discriminator

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
