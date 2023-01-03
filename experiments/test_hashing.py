import json
import logging
import os
import time
from datetime import datetime

import hydra.utils
import torch.nn.functional as F
import yaml
from omegaconf import DictConfig, OmegaConf

import engine
from utils import io
from utils.hashing import calculate_mAP, calculate_pr_curve


class RetrievalEvaluation:
    def __init__(self, config: DictConfig):

        io.init_save_queue()

        start_time = time.time()
        engine.seeding(config['seed'])

        logdir = config.logdir
        logging.info(yaml.dump(OmegaConf.to_object(config)))

        if config.use_last:
            modelfn = 'last'
        else:
            modelfn = 'best'

        trainer = hydra.utils.instantiate(config.trainer, config)
        trainer.load_dataset(load_db=True)
        trainer.load_dataloader()
        if config.exp not in ['descriptor', 'extract']:
            trainer.load_for_inference(logdir)
        trainer.load_model()
        trainer.load_criterion()
        if config.exp not in ['descriptor', 'extract']:
            trainer.load_model_state(f'{logdir}/models/{modelfn}.pth')
        trainer.to_device()

        eval_logdir = config.eval_logdir
        os.makedirs(eval_logdir, exist_ok=True)
        yaml.dump(OmegaConf.to_object(config), open(os.path.join(eval_logdir, 'eval_config.yaml'), 'w+'))

        self.config = config
        self.trainer = trainer
        self.start_time = start_time
        self.eval_logdir = eval_logdir

    def main(self):
        print('Testing Start')

        res = {}

        test_meters, test_out = self.trainer.inference_one_epoch('test', True)
        db_meters, db_out = self.trainer.inference_one_epoch('db', True)

        for key in test_meters: res['test_' + key] = test_meters[key].avg
        for key in db_meters: res['db_' + key] = db_meters[key].avg

        all_codes_names = []
        postfixes = []

        for key in test_out:
            if 'codes' in key:
                postfix = '_'.join(key.split('_')[1:])

                all_codes_names.append(key)
                postfixes.append(postfix)

        if self.config.exp != 'extract':
            if self.config.compute_mAP:
                for postfix, codes_name in zip(postfixes, all_codes_names):
                    print(f'Evaluating for "{codes_name}"')

                    db_labels = db_out['labels'].clone()
                    test_labels = test_out['labels'].clone()

                    if len(db_labels.size()) == 1:
                        db_labels = F.one_hot(db_labels, self.config.dataset.nclass)
                        test_labels = F.one_hot(test_labels, self.config.dataset.nclass)

                    mAPs, recalls, precisions = calculate_mAP(db_out[codes_name], db_labels,
                                                              test_out[codes_name], test_labels,
                                                              self.config.R,
                                                              threshold=self.config.ternary_threshold,
                                                              dist_metric=self.config.dist_metric,
                                                              PRs=self.config.PRs)
                    res['mAP' + postfix] = mAPs
                    res['recalls' + postfix] = recalls
                    res['precisions' + postfix] = precisions

                    if not isinstance(mAPs, list):
                        print(f'mAP@{self.config.R}: {mAPs:.4f}')
                    else:
                        for R, mAP in zip(self.config.R, mAPs):
                            print(f'mAP@{R}: {mAP:.4f}')

                    for R, recall, precision in zip(self.config.PRs, recalls, precisions):
                        print(f'P@{R}: {precision:.4f}; R@{R}: {recall:.4f}')
                    print()
            else:
                for postfix, codes_name in zip(postfixes, all_codes_names):
                    print(f'Evaluating for "{codes_name}"')
                    db_labels = db_out['labels'].clone()
                    test_labels = test_out['labels'].clone()

                    recalls, precisions, Rs = calculate_pr_curve(db_out[codes_name], db_labels,
                                                                 test_out[codes_name], test_labels,
                                                                 threshold=self.config.ternary_threshold,
                                                                 dist_metric=self.config.dist_metric)
                    res['recalls' + postfix] = recalls
                    res['precisions' + postfix] = precisions

                    for R, recall, precision in zip(Rs, recalls, precisions):
                        print(f'P@{R}: {precision:.4f}; R@{R}: {recall:.4f}')
                    print()

            json.dump(res, open(self.eval_logdir + '/history.json', 'w+'))

        if self.config.save_code or self.config.exp == 'extract':
            print('Saving code')
            io.fast_save({'test': test_out, 'db': db_out}, self.eval_logdir + '/outputs.pth')

        total_time = time.time() - self.start_time
        print(f'Testing End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Total time used: {total_time / (60 * 60):.2f} hours')
        print('Waiting for save queue to end')
        io.join_save_queue()
        print(f'Done: {self.eval_logdir}')
