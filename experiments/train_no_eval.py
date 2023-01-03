import logging
import time
from datetime import datetime

import numpy as np

from experiments.train_helper import RetrievalExperiment
from utils import io


class GeneralExperiment(RetrievalExperiment):

    def evaluation(self, ep, trainer, config):
        res = {'ep': ep + 1}

        test_meters, test_out = trainer.inference_one_epoch('test', True)

        for key in test_meters: res['test_' + key] = test_meters[key].avg

        return res, test_out

    def _load_best_stats(self):
        # infer curr_metric, best, best_ep from test_history
        self.curr_metric = self.test_history[-1]['test_loss']

        hist_losses = [res['test_loss'] for res in self.test_history]
        hist_eps = [res['ep'] for res in self.test_history]
        best_index = np.argmin(hist_losses)

        self.best = hist_losses[best_index]
        self.best_ep = hist_eps[best_index]

        # ep is checkpoint epoch, which is already added by 1
        # at run time, after this checkpoint, ep is added by 1
        # so it is correct to continue at epoch ((ep - 1) + 1)
        self.start_ep = self.train_history[-1]['ep']

    def main(self):
        ##### main loop: start learning for the compact representation #####
        self.best = 1e10

        for ep in range(self.start_ep, self.nepochs):
            res = {'ep': ep + 1}
            lrs = self.trainer.get_learning_rate()
            lrline = ''
            for i, lr in enumerate(lrs):
                lrline += f'{lr:.6f}; '
                res[f'lr/{i}'] = lr
            logging.info(f'Epoch [{ep + 1}/{self.nepochs}]; LR: {lrline.strip()[:-1]}')

            self.trainer.current_epoch = ep
            train_meters = self.trainer.train_one_epoch(ep=ep)

            for key in train_meters: res['train_' + key] = train_meters[key].avg
            self.record_history('train', res)

            eval_now = (ep + 1) == self.nepochs or (self.neval != 0 and (ep + 1) % self.neval == 0)
            if eval_now:
                res, test_out = self.evaluation(ep, self.trainer, self.config)
                self.curr_metric = res['test_loss']
                self.record_history('test', res)

                if self.best > self.curr_metric:  # loss is >
                    self.best = self.curr_metric
                    self.best_ep = ep + 1
                    self.trainer.save_model_state(f'{self.logdir}/models/best.pth')
                    self.trainer.save_codes(test_out, f'{self.logdir}/outputs/test_best.pth')
                self.trainer.save_codes(test_out, f'{self.logdir}/outputs/test_last.pth')

            save_now = self.nsave != 0 and (ep + 1) % self.nsave == 0
            if save_now:
                self.trainer.save_model_state(f'{self.logdir}/models/ep{ep + 1}.pth')

                if self.config.save_training_state:
                    self.trainer.save_training_state(f'{self.logdir}/optims/ep{ep + 1}.pth')

            self.trainer.save_model_state(f'{self.logdir}/models/last.pth')
            if self.config.save_training_state:
                self.trainer.save_training_state(f'{self.logdir}/optims/last.pth')
            self.wandb_commit()

        total_time = time.time() - self.start_time
        io.join_save_queue()

        logging.info(f'Training End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
        logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
        logging.info(f'Best mAP: {self.best:.6f} at {self.best_ep}')
        logging.info(f'Done: {self.logdir}')
