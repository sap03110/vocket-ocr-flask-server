import os
import cv2
import numpy as np
import pandas as pd
import shutil

def load_files():
    txt_path = "./text_detection/result"

    img_bbox = []
    for (dirpath, dirnames, filenames) in os.walk(txt_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm' and filename[
                                                                                                             :] == "res_androidFlask":
                img = cv2.imread(os.path.join(dirpath, file))
                cv2.imwrite("./result.jpg", img)
            if ext == '.txt':
                img_bbox.append(os.path.join(dirpath, file))

            #shutil.copy("./text_detection/result/res_androidFlask.jpg","./result.jpg")
    img_path = "./text_detection/test"

    img_files = []

    for (dirpath, dirnames, filenames) in os.walk(img_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
    return img_files, img_bbox


def crop_img(img_files, img_bbox):
    for n in range(len(img_files)):
        bbox = img_bbox[n]
        img_file = img_files[n]
        img = cv2.imread(img_file)
        txt = pd.read_csv(bbox, header=None)
        txt = txt.to_numpy()
        for num, i in enumerate(txt):
            a, b, c, d = i[:2], i[2:4], i[4:6], i[6:]

            poly = np.array((a, b, c, d))
            rect = cv2.boundingRect(np.abs(poly))

            croped = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].copy()
            cv2.imwrite("./text_recognition/demo_image/{}.jpg".format(num), croped)
