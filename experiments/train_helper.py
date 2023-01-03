import json
import logging
import os
import time
from datetime import datetime

import hydra.utils
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import normalized_mutual_info_score

import engine
from utils import io
from utils.hashing import calculate_mAP
from utils.logger import setup_logging, wandb_log, wandb_commit
from utils.machine_stats import print_stats


def check_keys(method, keys):
    for key in keys:
        if key in method:
            return True
    return False


def inference_one_batch_for_gldv2(self, *args, **kwargs):
    device = self.device

    data, meters = args
    image, labels, index = data
    image, labels = image.to(device), labels.to(device)

    arch_name = self.config['arch']

    with torch.no_grad():
        if check_keys(arch_name, ['cibhash']):
            feats = self.model.backbone(image)
            codes = self.model.encoder(feats)
        elif arch_name == '':  # shallow methods
            feats = image
            codes = self.model(feats)
        else:
            feats, codes = self.model(image)[:2]

        ids = index[0]

    return {
        'codes': codes.cpu(),
        'labels': labels.cpu(),
        'id': ids
    }


class RetrievalExperiment:
    def __init__(self, config: DictConfig):
        io.init_save_queue()

        start_time = time.time()
        engine.seeding(config.seed)

        logdir = config.logdir
        assert logdir != '', 'please input logdir'

        os.makedirs(logdir, exist_ok=True)
        setup_logging(logdir + '/log.txt')

        self.config = config
        self.logdir = logdir
        self.resume_logdir = config.resume_logdir
        self.resume = config.resume_logdir is not None
        self.wandb = config.wandb
        self.wandb_setup()

        print_stats()
        logging.info(yaml.dump(OmegaConf.to_object(config)))

        os.makedirs(f'{logdir}/models', exist_ok=True)
        os.makedirs(f'{logdir}/optims', exist_ok=True)
        os.makedirs(f'{logdir}/outputs', exist_ok=True)

        trainer = hydra.utils.instantiate(config.trainer, config)

        if self.resume:
            try:
                trainer.load_for_inference(self.resume_logdir)
            except:
                logging.warning('Not able to load for inference')

        trainer.prepare_before_training()
        trainer.save_before_training(logdir)

        trainer.load_dataset()
        trainer.load_dataloader()
        trainer.load_model()

        if config.finetune_path is not None:
            logging.info('Setup for fine-tuning')
            trainer.finetune_setup(config.finetune_path)

        trainer.load_optimizer_and_scheduler()
        trainer.load_criterion()

        train_history = []
        test_history = []

        best = 0
        best_ep = 0
        curr_metric = 0

        nepochs = config.epochs
        neval = config.eval_interval
        nsave = config.save_interval

        ##### setup variables #####
        self.train_history = train_history
        self.test_history = test_history
        self.best = best
        self.best_ep = best_ep
        self.curr_metric = curr_metric
        self.nepochs = nepochs
        self.neval = neval
        self.nsave = nsave
        self.trainer = trainer
        self.start_time = start_time
        self.start_ep = 0

        if self.resume:
            logging.info('Resume Start')
            self.resume_training()
        else:
            logging.info('Training Start')

        trainer.prepare_before_first_epoch()
        trainer.to_device()
        logging.info(repr(trainer.model))

    def wandb_setup(self):
        if self.wandb:
            assert 'WANDB_KEY' in os.environ, 'WANDB_KEY is missing.'
            logging.info('WANDB enabled.')
            wandb.login(key=os.environ['WANDB_KEY'])
            cfg = OmegaConf.to_object(self.config)
            wandb.init(name=self.logdir.split('logs/')[1], config=cfg)

    def wandb_commit(self):
        if self.wandb:
            wandb_commit()

    def record_history(self, stage, stats):
        if stage == 'train':
            history = self.train_history
        else:
            history = self.test_history

        if self.wandb:
            wandb_log({f'{stage}/{k}': v for k, v in stats.items()})

        history.append(stats)
        json.dump(history,
                  open(f'{self.logdir}/{stage}_history.json', 'w+'),
                  indent=True)

    def _load_best_stats(self):
        # infer curr_metric, best, best_ep from test_history
        if len(self.test_history) != 0:
            self.curr_metric = self.test_history[-1]['mAP']

            hist_mAPs = [res['mAP'] for res in self.test_history]
            hist_eps = [res['ep'] for res in self.test_history]
            best_index = np.argmax(hist_mAPs)

            self.best = hist_mAPs[best_index]
            self.best_ep = hist_eps[best_index]

        # ep is checkpoint epoch, which is already added by 1
        # at run time, after this checkpoint, ep is added by 1
        # so it is correct to continue at epoch ((ep - 1) + 1)
        self.start_ep = self.train_history[-1]['ep']

    def resume_training(self):
        if not os.path.exists(f'{self.resume_logdir}/train_history.json'):
            logging.warning('Training history not found, will not load training states')
            return

        # load history
        self.train_history = json.load(open(f'{self.resume_logdir}/train_history.json'))
        if os.path.exists(f'{self.resume_logdir}/test_history.json'):
            self.test_history = json.load(open(f'{self.resume_logdir}/test_history.json'))

        self._load_best_stats()

        # load optim state
        # todo: resume at any epoch, current implementation only for last epoch
        self.trainer.load_model_state(f'{self.resume_logdir}/models/last.pth')
        self.trainer.load_training_state(f'{self.resume_logdir}/optims/last.pth')

    def evaluation(self, ep, trainer, config):
        landmark_gt = None
        if self.config.dataset_name in ['gldv2', 'gldv2_delg']:
            trainer.__class__.inference_one_batch = inference_one_batch_for_gldv2
            landmark_gt_path = os.path.join(trainer.dataloader['test'].dataset.root, 'ground_truth.csv')
            landmark_gt = pd.read_csv(landmark_gt_path)  # id = index id, images = images id in database

        res = {'ep': ep + 1}

        test_meters, test_out = trainer.inference_one_epoch('test', True, ep=ep)
        db_meters, db_out = trainer.inference_one_epoch('db', True, ep=ep)

        for key in test_meters: res['test_' + key] = test_meters[key].avg
        for key in db_meters: res['db_' + key] = db_meters[key].avg

        all_codes_names = []
        postfixes = []

        for key in test_out:
            if 'codes' in key:
                postfix = '_'.join(key.split('_')[1:])

                all_codes_names.append(key)
                postfixes.append(postfix)

        assert len(all_codes_names) >= 1

        for postfix, codes_name in zip(postfixes, all_codes_names):
            logging.info(f'Evaluating for "{codes_name}"')
            if self.config.dataset_name in ['gldv2', 'gldv2_delg']:
                # to cpu here as calculate_mAP needs many GPU memory for gldv2
                db_out[codes_name] = db_out[codes_name].cpu()
                test_out[codes_name] = test_out[codes_name].cpu()

            mAP, recalls, precisions = calculate_mAP(db_out[codes_name], db_out['labels'],
                                                     test_out[codes_name], test_out['labels'],
                                                     config.dataset.R, dist_metric=config.dist_metric,
                                                     PRs=[1, 5, 10], landmark_gt=landmark_gt,
                                                     db_id=db_out.get('id'),
                                                     test_id=test_out.get('id'),
                                                     multiclass=self.config.dataset.multiclass)
            res['mAP' + postfix] = mAP
            res['recalls' + postfix] = recalls
            res['precisions' + postfix] = precisions

            logging.info(f'mAP: {res["mAP" + postfix]:.6f}')
            logging.info(f'R@10: {res["recalls" + postfix][-1]:.6f}')
            logging.info(f'P@10: {res["precisions" + postfix][-1]:.6f}')

        if 'pseudo_labels' in db_out and 'pseudo_labels' in test_out:
            db_nmi = normalized_mutual_info_score(db_out['labels'].argmax(1).cpu().numpy(),
                                                  db_out['pseudo_labels'].cpu().numpy())
            logging.info(f'DB NMI: {db_nmi:.4f}')
            test_nmi = normalized_mutual_info_score(test_out['labels'].argmax(1).cpu().numpy(),
                                                    test_out['pseudo_labels'].cpu().numpy())
            logging.info(f'Test NMI: {test_nmi:.4f}')
            res['db_nmi'] = db_nmi
            res['test_nmi'] = test_nmi

        return res, test_out, db_out

    def main(self):
        ##### main loop: start learning for the compact representation #####

        for ep in range(self.start_ep, self.nepochs):
            res = {'ep': ep + 1}
            lrs = self.trainer.get_learning_rate()
            lrline = ''
            for i, lr in enumerate(lrs):
                lrline += f'{lr:.6f}; '
                res[f'lr/{i}'] = lr
            logging.info(f'Epoch [{ep + 1}/{self.nepochs}]; LR: {lrline.strip()[:-1]}')  # [:-1] remove last ";"

            self.trainer.current_epoch = ep
            train_meters = self.trainer.train_one_epoch(ep=ep)

            for key in train_meters: res['train_' + key] = train_meters[key].avg
            self.record_history('train', res)

            eval_now = (ep + 1) == self.nepochs or (self.neval != 0 and (ep + 1) % self.neval == 0)
            if eval_now:
                res, test_out, db_out = self.evaluation(ep, self.trainer, self.config)
                self.curr_metric = res['mAP']
                self.record_history('test', res)

                if self.best < self.curr_metric:
                    self.best = self.curr_metric
                    self.best_ep = ep + 1
                    self.trainer.save_model_state(f'{self.logdir}/models/best.pth')
                    self.trainer.save_codes(db_out, f'{self.logdir}/outputs/db_best.pth')
                    self.trainer.save_codes(test_out, f'{self.logdir}/outputs/test_best.pth')

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
