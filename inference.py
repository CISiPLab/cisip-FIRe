import argparse

from inference.web import get_web_app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help="trained log path", type=str, required=True)
    parser.add_argument('--device', help="pytorch device, e.g. cuda:0, cpu", type=str, default='cpu')
    parser.add_argument('--k', help="retrieve first top k", type=int, default=10)
    args = parser.parse_args()

    app = get_web_app(args.dir, args.device, args.k)
    app.run(host='0.0.0.0', port=8000, debug=True)
