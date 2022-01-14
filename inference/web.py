import base64
import io
import os
import time
from zipfile import ZipFile

import numpy as np
from flask import Flask, render_template, request, flash, make_response, send_from_directory

from inference.indexer import Indexer
from utils.misc import pil_loader

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'webp', 'tif', 'jfif'}


def allowed_format(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_query(f, indexer: Indexer):
    assert indexer, 'Indexer not initialize'
    start_time = time.time()
    img_ = pil_loader(f)

    dist, ind, query_code = indexer.query_with_image(img_)
    img_paths = indexer.get_img_path(ind)

    end_time = time.time()
    flash("Upload successfully.")

    data = io.BytesIO()
    img_.save(data, "JPEG")

    encoded_img_data = base64.b64encode(data.getvalue())  # convert to base64 in byte
    encoded_img_data = encoded_img_data.decode('utf-8')  # convert to base64 in utf-8

    time_string = f'Time taken: {(end_time - start_time):.3f}s\n'
    code_string = "".join(str(np.unpackbits(query_code)).split())[1:-2]
    code_string = '\n'.join(code_string[i:i + 8] for i in range(0, len(code_string), 8))
    return dist, img_paths, code_string, encoded_img_data, img_, time_string


def get_web_app(log_path, device='cpu', top_k=10):
    indexer = Indexer(log_path, device=device, top_k=top_k)
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.secret_key = 'my_secret_key'

    @app.route('/', methods=['GET'])
    def index():
        return render_template('main.html')

    @app.route('/', methods=['POST'])
    def predict():
        f = request.files['file']
        if f.filename == '':
            flash("Please select a file")
            return index()
        elif not allowed_format(f.filename):
            flash("Invalid file type")
            return index()

        dist, img_paths, code_string, encoded_img_data, img_, time_string = process_query(f, indexer)

        return render_template('main.html', dists=dist[0], paths=img_paths[0],
                               code=code_string,
                               query_img=encoded_img_data,
                               query_img_full=img_,
                               time_string=time_string,
                               extra_data=indexer.get_info())

    @app.route('/zip', methods=['POST'])
    def generate_zip():
        f = request.files['file']
        try:
            img_ = pil_loader(f)
        except:
            flash("Invalid file type")
            return index()
        dists, ind, query_code = indexer.query_with_image(img_)
        img_paths = indexer.get_img_path(ind)
        dists = dists[0]

        data = io.BytesIO()
        img = io.BytesIO()
        img_.save(img, "JPEG")
        with ZipFile(data, 'w') as zip:
            for i, path in enumerate(img_paths[0]):
                zip.write(os.path.abspath(path),
                          f'retr_rank{i}_{dists[i]}_{path.split("/")[-1].split("_")[0]}.{path.split(".")[-1]}')
            zip.writestr(f'query.jpg', img.getvalue())
        data.seek(0)
        response = make_response(data.read())
        response.headers.set('Content-Type', 'zip')
        response.headers.set('Content-Disposition', 'attachment', filename='%s.zip' % 'out')
        return response

    @app.route('/img/<path:filename>')
    def get_image(filename):
        path, file = os.path.split(filename)
        print(os.path.abspath(filename))
        return send_from_directory(os.path.abspath(path), file)

    return app
