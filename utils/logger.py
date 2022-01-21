import logging


def setup_logging(filename):
    stream_handler = logging.StreamHandler()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=filename,
                        filemode='a',
                        level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S')
    stream_handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s: %(message)s', '%d-%m-%y %H:%M:%S'))
    logging.getLogger().addHandler(stream_handler)


def setup_logging_display_only():
    stream_handler = logging.StreamHandler()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(asctime)s: %(message)s',
                        datefmt='%d-%m-%y %H:%M:%S')
    stream_handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s: %(message)s', '%d-%m-%y %H:%M:%S'))
    logging.getLogger().addHandler(stream_handler)
