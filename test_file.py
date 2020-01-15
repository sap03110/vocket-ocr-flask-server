import sys

sys.path.append("./Mask_RCNN")
import cv2
import main
import fire
import json
import os
import csv
import pprint
import werkzeug
from waitress import serve
from flask import Flask, request, send_file
from flask_json import json_response, as_json
from flask_cors import CORS
from cachetools import cached, TTLCache
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN
from numba import cuda

"""
"""
import pandas as pd
from file_util import load_files, crop_img

import text_recognition.demo as recognition
import text_detection.test as detection


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "underline_cfg"

    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Do not use the development server in a production environment.
# Create the application instance
app = Flask(__name__)
CORS(app)
app.config.from_object(__name__)  # Load config from app.py file
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['MAX_CONTENT_LENGTH'] = (1024 * 1024) * 25  # 5MB
app.config['ENV_DEFAULT_PORT'] = "8000"
app.config['ENV_DEBUG_MODE'] = True
app.config['JSON_ADD_STATUS'] = True
app.config['JSON_STATUS_FIELD_NAME'] = 'server_status'
app.config['JSON_JSONP_OPTIONAL'] = False
app.config['JSON_DECODE_ERROR_MESSAGE'] = True
app.config['Threaded'] = True
app.config['APP_HOST_NAME'] = '0.0.0.0'

cache = TTLCache(maxsize=300, ttl=360)


def response_json_ops(custom_status=200, status=200, res_msg='Put on a happy face'):
    return json_response(server_status=custom_status, status_=status, message=res_msg)


def server_ops():
    p = pprint.PrettyPrinter(indent='4')
    # ------
    # Display server information :)
    p.pprint(app.config)
    # To allow aptana to receive errors, set use_debugger=False
    # app.run(port=app.config['ENV_DEFAULT_PORT'], debug=app.config['ENV_DEBUG_MODE'])
    # Deploy Server with Web Serve Gateway Interface
    serve(app=app, host=app.config['APP_HOST_NAME'], port=app.config['ENV_DEFAULT_PORT'])


@app.route('/aa', methods=["POST"])  # 전체인식
@as_json
def test_api_post2():
    path = "C:\\Users\\CAU\\Desktop\\capstone\\text_recognition\demo_image"
    if os.path.exists(path):
        for file in os.scandir(path):
            os.remove(file.path)

    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save("./text_detection/test/" + filename)
    detection.run_detection()

    img_files, img_bbox = load_files()
    crop_img(img_files, img_bbox)
    pred_str = recognition.run_recognition()

    for i, file in enumerate(img_files):
        txt = pd.read_csv(img_bbox[i], header=None)
        df = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "result_text"])

        for num, _col in enumerate(list(df)[:-1]):
            df[_col] = txt[num]
        df["result_text"] = pred_str
        df.to_csv("./result.csv")
    return "done"


@app.route('/', methods=["GET"])
def test_api_get():
    response_header = response_json_ops()
    return send_file('./result.jpg')


@app.route('/test', methods=["GET", "POST"])
def ttest():
    d = []
    with open('./result.csv', 'r') as f:
        reader = csv.DictReader(f)
        for c in reader:
            dd = {}
            for k, v in c.items():
                if k != "" and k != "x2" and k != "y2" and k != "x4" and k != "y4":
                    if v not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
                        dd[k] = v
            d.append(dd)
    djson = json.dumps(d)
    return djson


@cached(cache)
def read_data():
    main.run_main()


@app.route('/bb', methods=["POST"])  # 밑줄인식, 용량 문제로 github 업로드가 안됨
@as_json
def test_api_post():

    path = "C:\\Users\\CAU\\Desktop\\capstone\\text_recognition\demo_image"
    if os.path.exists(path):
        for file in os.scandir(path):
            os.remove(file.path)

    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save("./text_detection/test/" + filename)
    # time.sleep(5)
    detection.run_detection()
    # time.sleep(5)
    img_files, img_bbox = load_files()
    crop_img(img_files, img_bbox)
    pred_str = recognition.run_recognition()

    # underline detection
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model_path = 'mask_rcnn_underline_cfg_0020.h5'
    model.load_weights(model_path, by_name=True)
    temp = cv2.imread("./text_detection/test/androidFlask.jpg")

    yhat = model.detect([temp], verbose=0)[0]
    print(len(yhat['rois']))
    # [l, t], [r, t], [r, b], [l, b]
    for i, file in enumerate(img_files):
        txt = pd.read_csv(img_bbox[i], header=None)
        df = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "result_text"])
        # compare

        for i, bb in enumerate(txt.values):
            x1, y1, x2, y2, x3, y3, x4, y4 = bb
            # textbb = [x1, y1, x3, y3]
            for underline in yhat['rois']:
                uy1, ux1, uy2, ux2 = underline
                if (ux1 + ux2) / 2 > x1 and (ux1 + ux2) / 2 < x3 and y1 < uy1 and uy1 < y3:
                    df = df.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3, "x4": x4, "y4": y4,
                                    "result_text": pred_str[i]}, ignore_index=True)
                    temp = cv2.rectangle(temp, (x1, y1), (x3, y3), (0, 0, 255), 3)
                # top-left corner and bottom-right corner of rectangle.

    df.to_csv("./result.csv")
    cv2.imwrite("./result.jpg", temp)
    from keras import backend as K
    K.clear_session()
    cuda.select_device(0)
    cuda.close()

    del model
    return "done"


if __name__ == '__main__':
    fire.Fire(server_ops)
