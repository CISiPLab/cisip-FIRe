import logging
import sys

import wandb

cache = {}


def setup_logging(filename):
    stream_handler = logging.StreamHandler(sys.stdout)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=filename,
                        filemode='a',
                        level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S')
    stream_handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s: %(message)s', '%d-%m-%y %H:%M:%S'))
    logging.getLogger().addHandler(stream_handler)


def setup_logging_stream_only():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S')


def wandb_log(data):
    global cache
    cache.update(data)


def wandb_commit():
    global cache

    wandb.log(cache, commit=True)
    cache = {}
