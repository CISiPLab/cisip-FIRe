import json
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn

import configs
from functions.ternarization import tnt
from scripts.train_helper import prepare_model


def prepare_model_simple(model_dir):
    cfg = json.load(open(model_dir + '/config.json'))

    model = prepare_model(cfg, torch.device('cpu'))[0]
    return model


def prepare_dataset_simple(dataset, filename,
                           resize=256,
                           crop=224):
    # simplify the dataset preparation
    cfg = {
        'dataset': dataset,
        'dataset_kwargs': {
            'resize': resize,
            'crop': crop
        },
        'arch_kwargs': {
            'nclass': configs.nclass(dataset)
        }
    }
    return configs.dataset(cfg, filename, 'test')


def get_ternarization_config(mode='tnt', threshold=0):
    assert mode in ['tnt', 'threshold'], f'invalid mode: {mode}'
    assert threshold >= 0, 'threshold < 0'
    return {
        'mode': mode,
        'threshold': threshold
    }


class Retrieval:
    # todo: fine-grained retrieval --> retrieve by hash code, then by features

    def __init__(self, model, database, device,
                 ternarization=None,
                 distance_func='hamming',
                 re_rank=False,
                 layers_for_global_retrieval=['hash_fc']):
        """
        model: model
        database: Dataset object
        device: torch.device
        ternarization: a dictionary about ternarization config
        distance_func: 'hamming' / 'euclidean'

        """
        self.model = model.to(device)
        self.database = database
        self.device = device
        self.ternarization = ternarization
        self.layers_for_global_retrieval = layers_for_global_retrieval
        self.distance_func = distance_func
        self.re_rank = re_rank  # todo:
        self.hash_db = True  # not self.re_rank and self.distance_func == 'hamming'

        self.forward_cache = {}
        self.forward_hooks = []

        self.setup_forward_hook()

        self.database_features = None
        if self.hash_db:
            self.database_codes = self.preprocess()
        else:
            self.database_codes, self.database_features = self.preprocess()
        self.collection = None

    def setup_forward_hook(self):
        for layer_name in self.layers_for_global_retrieval:
            layer = eval(f'self.model.{layer_name}')  # type: nn.Module

            def save_output(module, inp, oup):
                # assume the module only hv 1 inp
                # this will direct save the output to the key, beware of how we handle forward_cache
                self.forward_cache[layer_name] = oup.data  # .data is similar to .detach()

            hook_handle = layer.register_forward_hook(save_output)
            self.forward_hooks.append(hook_handle)

    def remove_forward_hook(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.forward_hooks.clear()

    def preprocess(self):
        dataloader = configs.dataloader(self.database, bs=256, shuffle=False, workers=8, drop_last=False)

        codes = []
        feats = []
        for bidx, (data, label) in enumerate(dataloader):
            print(f'processing batch [{bidx}/{len(dataloader)}]', end='\r')
            data = data.to(self.device)
            with torch.no_grad():
                # old method
                # code_logits = self.model(data)[1]
                # if not self.hash_db:
                #     feats.append(code_logits.clone())  # must clone, or else to_hash will mutate the features
                # code_logits = self.to_hash(code_logits)

                # new method
                # forward
                self.forward_cache.clear()
                self.model(data)

                code_logits = []
                for layer_name in self.forward_cache:
                    layer_output = self.forward_cache[layer_name]
                    code_logits.append(layer_output.view(layer_output.size(0), -1))  # resize to (BS, size)
                code_logits = torch.cat(code_logits, dim=1)  # (BS, size + ...)
                if self.distance_func == 'hamming':
                    code_logits = self.to_hash(code_logits)
                elif self.distance_func == 'cosine':
                    code_logits = code_logits / (code_logits.norm(p=2, dim=1, keepdim=True) + 1e-7)
            codes.append(code_logits)

        if self.hash_db:
            return torch.cat(codes)
        else:
            return torch.cat(codes), torch.cat(feats)

    def to_hash(self, code_logits):
        if self.ternarization is not None:
            mode = self.ternarization['mode']
            if mode == 'tnt':
                code_logits = tnt(code_logits)
            elif mode == 'threshold':
                threshold = self.ternarization['threshold']
                if threshold != 0:
                    # if value within margin, set to 0
                    code_logits[code_logits.abs() < threshold] = 0

        code_logits = code_logits.sign()
        return code_logits

    def compute_distance(self, queries, db):
        bs, nbit = queries.size()

        if self.distance_func == 'hamming':
            dist = 0.5 * (nbit - queries @ db.t())
        elif self.distance_func == 'euclidean':
            dist = torch.cdist(queries, db, p=2)
        elif self.distance_func == 'cosine':
            dist = 1. - queries @ db.t()  # both should be already in unit vec
        else:
            raise NotImplementedError(f'invalid distance func: {self.distance_func}')

        return dist

    def retrieve(self, query, topk=1000, image=True, label=False, distance=False, codes=False):
        """
        query: (b*3*h*w)
        return retrieved images
        """
        self.forward_cache.clear()
        self.model(query)

        query_codes = []
        for layer_name in self.forward_cache:
            layer_output = self.forward_cache[layer_name]
            query_codes.append(layer_output.view(layer_output.size(0), -1))  # resize to (BS, size)
        query_codes = torch.cat(query_codes, dim=1)

        self.forward_cache.clear()  # clear after forward to save memory
        if self.distance_func == 'hamming':
            query_codes = self.to_hash(query_codes)
        elif self.distance_func == 'cosine':
            query_codes = query_codes / (query_codes.norm(p=2, dim=1, keepdim=True) + 1e-7)

        # query_features = self.model(query)[1]
        # query_codes = self.to_hash(query_features)

        bs, nbit = query_codes.size()

        dist = self.compute_distance(query_codes, self.database_codes)
        # idxs = torch.argsort(dist, 1, descending=False)
        idxs = torch.topk(dist, k=topk, dim=1, largest=False)[1]

        retrieved = [[] for _ in range(bs)]
        for q in range(bs):
            for i in idxs[q, :topk]:
                ret = {}
                if image or label:
                    img, lbl = self.database[i]
                    if image:
                        ret['image'] = img
                    if label:
                        ret['label'] = lbl

                if distance:
                    ret['distance'] = dist[q, i].cpu()

                if codes:
                    ret['codes'] = {
                        'database': self.database_codes[i].cpu(),
                        'query': query_codes[q].cpu()
                    }

                retrieved[q].append(ret)

        return retrieved

    def request_intermediate_output(self, output_layers: List[str]):
        """
        Collect the layer outputs from the intermediate layer.
        This method should be called before the retrieve()
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

    def __del__(self):
        self.remove_forward_hook()
