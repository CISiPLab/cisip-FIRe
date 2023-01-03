import torch

from trainers.orthohash import OrthoHashTrainer
from utils.hashing import get_hamm_dist
from utils.metrics import calculate_accuracy_hamm_dist


class DPNTrainer(OrthoHashTrainer):
    """
    DPN is similar to OrthoHash except the forward is different
    """

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            codes = self.model(image)
            loss = self.criterion(codes, labels)
            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))
            meters['hacc'].update(hacc.item(), image.size(0))

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        codes = self.model(image)
        loss = self.criterion(codes, labels)

        # backward and update
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
        meters['hacc'].update(hacc.item(), image.size(0))
