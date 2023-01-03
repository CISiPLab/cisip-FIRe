import torch

from trainers.base import BaseTrainer


class PairwiseTrainer(BaseTrainer):
    def __init__(self, config):
        super(PairwiseTrainer, self).__init__(config)

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            codes = self.model(image)
            loss = self.criterion(codes, labels, index)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels, index = image.to(device), labels.to(device), index.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        codes = self.model(image)
        loss = self.criterion(codes, labels, index)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
