import logging

import hydra

from trainers.base_pairwise import PairwiseTrainer


class HashNetTrainer(PairwiseTrainer):

    def load_criterion(self):
        self.criterion = hydra.utils.instantiate(self.config.criterion, train_size=len(self.dataset['train']))

    def pre_epoch_operations(self):
        ep = self.current_epoch
        step_continuation = self.config['method']['param']['step_continuation']
        self.criterion.beta = (ep // step_continuation + 1) ** 0.5
        logging.info(f'Updated Beta At Epoch {ep + 1}: {self.criterion.beta}')

    def train_one_epoch(self, **kwargs):
        self.pre_epoch_operations()
        return super(HashNetTrainer, self).train_one_epoch(**kwargs)
