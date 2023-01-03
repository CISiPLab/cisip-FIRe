from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from trainers.base import BaseTrainer


class DescriptorTrainer(BaseTrainer):

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data = args[0]
        image, labels, index = data
        image = image.to(device)

        with torch.no_grad():
            codes = self.model(image)

        return {
            'codes': codes.cpu(),
            'labels': labels
        }

    def inference_one_epoch(self, datakey='test', return_codes=False, **kwargs):
        self.model.eval()

        ret = defaultdict(list)
        size = len(self.dataloader[datakey])
        iterator = iter(self.dataloader[datakey])

        with tqdm(range(size)) as tepoch:
            for i in tepoch:
                data = next(iterator)
                output = self.inference_one_batch(data, bidx=i, **kwargs)

                for key in output:
                    ret[key].append(output[key])

        res = {}
        for key in ret:
            if isinstance(ret[key][0], torch.Tensor):
                res[key] = torch.cat(ret[key])
            else:
                res[key] = np.concatenate(ret[key])

        return {}, res
