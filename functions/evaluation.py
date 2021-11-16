import json
import logging
import os
from collections import defaultdict
from typing import List

import torch

import configs
from functions.metrics import calculate_mAP
from scripts.train_general import test_hashing as classify_test
from scripts.train_general import test_hashing as unsupervised_test
from scripts.train_helper import prepare_model, prepare_dataloader
from constants import losses as LOSSES

POSSIBLE_OUTPUT_NAME = ['best', 'last', 'fc']


def get_output_filename(data_subset, name):
    choices = {
        "best": "best",
        "last": "out",
        "fc": "fc"
    }
    return f"{data_subset}_{choices[name]}"


class Evaluation:
    def __init__(self):
        self.model = None
        self.loss_name = None

    def load_queries(self):
        raise NotImplementedError()

    def load_database(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def get_test_function(self):
        if self.loss_name in LOSSES['classification']:
            test_func = classify_test
        elif self.loss_name in LOSSES['unsupervised']:
            test_func = unsupervised_test
        # TODO: delg, autoencoder, pairwise
        else:
            raise NotImplementedError(f'no implementation for {self.loss_name}')
        return test_func


class EvaluationHashing(Evaluation):
    """
    Evaluation on Hashing algorithms
    Create a hashing model evaluator, allow the model to evaluate on query and database set,
    and calculate for mAP score.
    log_path: model log path, which stored model outputs, history, weight, etc
    model_selection: best, last
    device: torch.device
    """

    def __init__(self, log_path, model_selection="best", device=torch.device('cuda:0')):
        self.log_path = log_path
        self.model_selection = model_selection
        self.config = json.load(open(log_path + '/config.json'))
        self.device = device
        configs.seeding(self.config['seed'])
        self.R = self.config['R']
        self.loss_name = self.config['loss']
        # setup model
        self.model = self.load_model()
        self.test_loader = self.db_loader = None
        self.queries = None
        self.database = None
        self.test_func = self.get_test_function()
        self.collection = None

    def load_model(self):
        model, _ = prepare_model(self.config, self.device)
        self.test_loader, self.db_loader = prepare_dataloader(self.config)[1:]
        self.model.load_state_dict(
            torch.load(f'{self.log_path}/models/{self.model_selection}.pth', map_location=self.device))
        if self.config["loss"] in ['dpn', 'orthocos', 'orthoarc']:
            self.model.centroids = torch.load(
                os.path.join(self.log_path, "outputs/centroids.pth"), map_location=self.device)
        return model

    def request_intermediate_output(self, output_layers: List[str]):
        """
        Collect the layer outputs from the intermediate layer.
        This method should be called before the load_queries or load_database
        output_layers: output layer that wanted to collect
        """
        self.collection = defaultdict(list)

        def outer(name):
            def get_output(self, input, output):
                self.collection[name].append(output.cpu())
            return get_output

        for layer in output_layers:
            eval(f'self.model.{layer}').register_forward_hook(outer(layer))

    def collect_intermediate_output(self):
        if self.collection:
            if len(self.collection) > 1:
                for key in self.collection:
                    self.collection[key] = torch.cat(self.collection[key])
        return self.collection


    def load_data(self, data_subset):
        path = os.path.join(self.log_path, f'/outputs/{get_output_filename(data_subset, self.model_selection)}.pth')
        if not os.path.isfile(path):
            _, data_out = self.test_func(self.model, self.test_loader, self.device, self.loss_name,
                                         self.config['loss_param'], return_codes=True)
            torch.save(data_out, path)
        else:
            data_out = torch.load(path, map_location=self.device)
        logging.info(f'Loaded {data_subset} set on {path}')
        return data_out

    def load_queries(self):
        test_out = self.load_data("test")
        self.queries = test_out
        return test_out

    def load_database(self, name):
        db_out = self.load_data("db")
        self.database = db_out
        return db_out

    def eval(self, distance_func="hamming", ternarization=None, shuffle_database=False):
        assert self.database is not None
        assert self.queries is not None
        map_result = calculate_mAP(self.database['codes'], self.database['labels'],
                                   self.queries['codes'], self.queries['labels'],
                                   self.R,  # does it need an option?
                                   ternarization=ternarization,
                                   distance_func=distance_func,
                                   shuffle_database=shuffle_database,
                                   device=self.device)
        return map_result
