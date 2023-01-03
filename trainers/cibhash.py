import torch

from trainers.base_contrastive import ContrastiveTrainer


class CIBHashTrainer(ContrastiveTrainer):

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            _, codes = self.model(images)

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images_i, images_j = images
        images_i, images_j, labels = images_i.to(device), images_j.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        prob_i, z_i = self.model(images_i)
        prob_j, z_j = self.model(images_j)
        loss = self.criterion(prob_i, prob_j, z_i, z_j)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item())
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item())
