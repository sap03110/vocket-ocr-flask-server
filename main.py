import pandas as pd
from file_util import load_files, crop_img
import time
import text_recognition.demo as recognition
import text_detection.test as detection
import sys
sys.path.append("./Mask_RCNN")
import cv2

from Mask_RCNN.mrcnn.config import Config


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "underline_cfg"

    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def run_main():
    detection.run_detection()

    img_files, img_bbox = load_files()
    crop_img(img_files, img_bbox)
    pred_str = recognition.run_recognition()

    print("recog done")

    from Mask_RCNN.mrcnn.model import MaskRCNN

    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model_path = 'mask_rcnn_underline_cfg_0019.h5'
    model.load_weights(model_path, by_name=True)

    temp = cv2.imread("./text_detection/test/androidFlask.jpg")

    yhat = model.detect([temp], verbose=0)[0]
    # [l, t], [r, t], [r, b], [l, b]
    for i, file in enumerate(img_files):
        txt = pd.read_csv(img_bbox[i], header=None)
        df = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "result_text"])
        # compare
        #temp = []
        for i, bb in enumerate(txt.values):
            x1, y1, _, _, x3, y3, _, _ = bb
            # textbb = [x1, y1, x3, y3]
            for underline in yhat['rois']:
                uy1, ux1, uy2, ux2 = underline
                if (ux1+ux2)/2 > x1 and (ux1+ux2)/2 < x3 and y1<uy1 and uy1<y3:
                    print(pred_str[i])



        for num, _col in enumerate(list(df)[:-1]):
            df[_col] = txt[num]
        df["result_text"] = pred_str
        df.to_csv("./result.csv")
