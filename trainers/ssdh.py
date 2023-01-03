import torch

from models.loss.ssdh import SemanticStructureDHLoss
from trainers.base import BaseTrainer


class SSDHTrainer(BaseTrainer):

    def prepare_before_first_epoch(self):
        self.criterion.compute_similarity(self.model,
                                          self.config,
                                          self.dataloader['train'])  # type: SemanticStructureDHLoss

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            feats, codes = self.model(image)
            # no need to do evaluation for ssdh

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        self.optimizer.zero_grad()

        feats, codes = self.model(image)
        loss = self.criterion(codes, index)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
