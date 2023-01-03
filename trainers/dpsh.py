import hydra

from trainers.base_pairwise import PairwiseTrainer


class DPSHTrainer(PairwiseTrainer):

    def load_criterion(self):
        self.criterion = hydra.utils.instantiate(self.config.criterion, train_size=len(self.dataset['train']))
