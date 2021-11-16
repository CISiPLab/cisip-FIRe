import argparse
import json

from scripts import test_hashing


def get_ternarization_config(args):
    return {
        'mode': args.tmode,
        'threshold': args.threshold
    }


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', required=True)
parser.add_argument('--db-path', default=None)
parser.add_argument('--test-path', default=None)
parser.add_argument('--old-eval', default=False, action='store_true', help='whether to use old eval method')
parser.add_argument('--tmode', default='off', choices=['tnt', 'threshold', 'off'], help='ternarization mode')
parser.add_argument('--threshold', default=0., type=float, help='threshold for ternary')
parser.add_argument('--dist', default='hamming', choices=['hamming', 'euclidean', 'cosine', 'jmlh-dist'])
parser.add_argument('--shuffle', default=False, action='store_true', help='whether to shuffle database')
parser.add_argument('--tag', default=None)
parser.add_argument('--device', default='cuda:0', type=str, help='cuda:x')
parser.add_argument('--R', default=0, type=int, help='0 = default, -1 = all')
parser.add_argument('--zero-mean-eval', default=False, action='store_true')

args = parser.parse_args()

logdir = args.logdir
config = json.load(open(logdir + '/config.json'))

config.update({
    'ternarization': get_ternarization_config(args),
    'distance_func': args.dist,
    'shuffle_database': args.shuffle,
    'db_path': logdir + '/' + str(args.db_path),
    'test_path': logdir + '/' + str(args.test_path),
    'load_model': args.db_path is None,
    'tag': args.tag,
    'old_eval': args.old_eval,
    'device': args.device,
    'zero_mean_eval': args.zero_mean_eval
})

if args.R != 0 and config['R'] != args.R:
    config['R'] = args.R

config['dataset_kwargs']['remove_train_from_db'] = False

test_hashing.main(config)
