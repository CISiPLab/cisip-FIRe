import json
import logging
import os

from PIL import Image
import faiss
import numpy as np
import torch

from scripts.test_hashing import get_codes
from scripts.train_helper import prepare_dataloader, prepare_model
from utils import io


class Indexer:
    def __init__(self, log_path, device='cpu', top_k=10):
        # attributes
        self.config = None
        self.img_paths = None
        self.index = None
        self.model = None
        self.nbit = None
        self.nclass = None
        self.transform = None
        self.device = torch.device(device)
        self.log_path = log_path
        assert os.path.exists(log_path), f'Log path not exists: {log_path}'
        os.makedirs(os.path.join(log_path, 'inference'), exist_ok=True)
        self.top_k = top_k

        # prepare
        io.init_save_queue()

        # actions
        self.read_log_path()
        self.load_model()
        if not os.path.exists(os.path.join(self.log_path, 'inference', 'data.pth')):
            logging.info(f"Inference data not yet created, creating now. This will take a while.")
            self.generate_inference_data()
        self.build_index()
        self.get_transform()

    def read_log_path(self):
        logging.info(f'Reading log path at {self.log_path}')
        self.config = json.load(open(self.log_path + '/config.json'))
        self.nbit = self.config['arch_kwargs']['nbit']
        self.nclass = self.config['arch_kwargs']['nclass']
        assert self.nbit % 8 == 0, 'Number of bit must be multiple of 8'

    def load_model(self):
        model_path = os.path.join(self.log_path, 'models', 'best.pth')
        assert os.path.exists(model_path), f'Model path not exists: {model_path}'
        assert self.config, f'Config did not load properly.'

        # load model
        self.model = prepare_model(self.config, self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_transform(self):
        _, db_loader = prepare_dataloader(self.config, include_train=False)[1:]  # test, database loaders
        self.transform = db_loader.dataset.transform

    def generate_inference_data(self):
        assert self.model, f'Model not loaded. Call Indexer.load_model() to load the model.'
        if os.path.exists(os.path.join(self.log_path, 'inference', 'data.pth')):
            logging.info(f"Inference Data already exists, recreate and overwrite it.")
        _, db_loader = prepare_dataloader(self.config, include_train=False)[1:]  # test, database loaders
        logging.info('DB Dataset ' + str(len(db_loader.dataset)))

        db_paths = db_loader.dataset.get_img_paths()
        db_out = get_codes(self.model, db_loader, self.device)

        out = self.convert_int(db_out['codes'])
        del db_out

        torch.save({
            'img_paths': db_paths,
            'img_hashes': out
        }, os.path.join(self.log_path, 'inference', 'data.pth'))

    def build_index(self):
        assert os.path.exists(os.path.join(self.log_path, 'inference', 'data.pth')),\
            f'No inference data. Consider generate it by calling Indexer.generate_inference_data()'
        inference_data = torch.load(os.path.join(self.log_path, 'inference', 'data.pth'))
        logging.info(f"Creating Index")
        self.index = faiss.IndexBinaryFlat(self.nbit)
        self.index.add(inference_data['img_hashes'])
        self.img_paths = inference_data['img_paths']

    @staticmethod
    def convert_int(codes):
        out = codes.sign().cpu().numpy().astype(int)
        out[out == -1] = 0
        out = np.packbits(out, axis=-1)
        return out

    def get_img_path(self, ind):
        return self.img_paths[ind]

    def query_with_code(self, query_code):
        assert self.index, f'Index not built. Consider build index by calling Indexer.build_index()'
        dist, ind = self.index.search(query_code, self.top_k)
        return dist, ind

    def query_with_image(self, query_img: Image.Image):
        assert self.img_paths is not None, 'Not loaded index.'
        assert self.model, 'Model not loaded'
        # preprocess
        img = self.transform(query_img)
        # extract code with trained model
        with torch.no_grad():
            codes = self.model(img.unsqueeze(0).to(self.device))[1]
            query_code = (codes.sign() + 1) / 2
        query_code = query_code.bool().cpu().numpy()
        query_code = np.packbits(query_code, axis=-1)
        dist, ind = self.query_with_code(query_code)
        return dist, ind, query_code

    def get_info(self):
        return {
            'dataset': self.config['dataset'],
            'number_of_class': self.nclass,
            'number_of_bits': self.nbit,
            'device': self.device,
            'log_path': self.log_path,
            'loss': self.config['loss']
        }
