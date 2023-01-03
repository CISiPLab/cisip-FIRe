import torch

from trainers.base import BaseTrainer


class ContrastiveTrainer(BaseTrainer):
    def __init__(self, config):
        super(ContrastiveTrainer, self).__init__(config)

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            codes = self.model(images)

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images_0, images_1 = images
        images_0, images_1, labels = images_0.to(device), images_1.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        codes_0 = self.model(images_0)
        codes_1 = self.model(images_1)
        loss = self.criterion(codes_0, codes_1, labels, index)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), images.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), images.size(0))
