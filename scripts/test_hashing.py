import json
import logging
import os
import time
from datetime import datetime

import torch
from tqdm import tqdm

import configs
from functions.metrics import calculate_mAP
from scripts.train_helper import prepare_dataloader, prepare_model
from utils import io
from utils.logger import setup_logging


def get_codes(model, test_loader, device):
    model.eval()

    ret_codes = []
    ret_labels = []

    pbar = tqdm(test_loader, desc='Test', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}',
                disable=configs.disable_tqdm)
    for i, (data, labels, index) in enumerate(pbar):
        with torch.no_grad():
            data, labels = data.to(device), labels.to(device)
            code_logits = model(data)[1]  # code logits always at position 1

            ret_codes.append(code_logits)
            ret_labels.append(labels)

    res = {
        'codes': torch.cat(ret_codes),
        'labels': torch.cat(ret_labels)
    }
    return res


def load_codes(path, device):
    return torch.load(path, map_location=device)


def main(config):
    device = torch.device(config['device'])
    io.init_save_queue()

    start_time = time.time()
    configs.seeding(config['seed'])

    logdir = config['logdir']

    result_logdir = logdir + '/testing_results'
    count = 0
    orig_logdir = result_logdir
    result_logdir = orig_logdir + f'/{count:03d}'
    while os.path.isdir(result_logdir):
        count += 1
        result_logdir = orig_logdir + f'/{count:03d}'
    os.makedirs(result_logdir, exist_ok=True)

    setup_logging(result_logdir + '/test_log.txt')
    logging.info(json.dumps(config, indent=2))

    if config['load_model']:
        logging.info('Load model')
    else:
        logging.info('Load codes')

    json.dump(config, open(os.path.join(result_logdir, 'eval_history.json'), 'w+'))

    logging.info('Testing Start')

    res = {}

    if config['load_model']:  # load model and get codes
        # load dataset
        test_loader, db_loader = prepare_dataloader(config, include_train=False)[1:]
        logging.info('Test Dataset ' + str(len(test_loader.dataset)))
        logging.info('DB Dataset ' + str(len(db_loader.dataset)))

        # load model
        model = prepare_model(config, device)
        model.load_state_dict(torch.load(f'{logdir}/models/best.pth', map_location=device))

        # get codes
        test_out = get_codes(model, test_loader, device)
        db_out = get_codes(model, db_loader, device)

        io.fast_save(test_out, result_logdir + '/test_outputs.pth')
        io.fast_save(db_out, result_logdir + '/db_outputs.pth')
    else:
        test_out = load_codes(config['test_path'], device)
        db_out = load_codes(config['db_path'], device)

        logging.info(f'DB: {db_out["codes"].size()}; Test: {test_out["codes"].size()}')

    # todo: not support for landmark dataset
    res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                               test_out['codes'], test_out['labels'],
                               config['R'],
                               ternarization=config['ternarization'],
                               distance_func=config['distance_func'],
                               shuffle_database=config['shuffle_database'],
                               device=device,
                               zero_mean=config['zero_mean_eval'])

    json.dump(res, open(result_logdir + '/history.json', 'w+'))

    total_time = time.time() - start_time
    logging.info(f'Testing End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'mAP: {res["mAP"]:.4f}')
    logging.info(f'Total time used: {total_time / (60 * 60):.2f} hours')
    logging.info('Waiting for save queue to end')
    io.join_save_queue()
    logging.info(f'Done: {result_logdir}')
