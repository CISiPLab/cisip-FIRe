import argparse

from inference.web import get_web_app
from utils.logger import setup_logging_display_only

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="trained log path", type=str, required=True)
    parser.add_argument('--device', help="pytorch device, e.g. cuda:0, cpu", type=str, default='cpu')
    parser.add_argument('--k', help="retrieve first top k", type=int, default=10)
    parser.add_argument('--ip', help='ip address', type=str, default='0.0.0.0')
    parser.add_argument('--port', help='tcp port', type=int, default=8000)
    parser.add_argument('--log', help='enable log', action='store_true')
    args = parser.parse_args()

    if args.log:
        setup_logging_display_only()

    app = get_web_app(args.dir, args.device, args.k)
    app.run(host=args.ip, port=args.port, debug=True)
