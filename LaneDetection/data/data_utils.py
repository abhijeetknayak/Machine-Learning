import numpy as np
import json
import cv2
from easydict import EasyDict as edict
from config import *

def lane_visualization(img, h_samples, lanes):
    lane_viz = [[(x, y) for (x, y) in zip(lane, h_samples) if x >= 0] for lane in lanes]
    for lane in lane_viz:
        cv2.polylines(img, np.int32([lane]), isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def load_json_data(path, name):
    with open(path + name) as json_file:
        data = json_file.readline()
        data = edict(json.loads(data))

        img = cv2.imread(path + data.raw_file)

        h_samples = data.h_samples
        lanes = data.lanes

        img = np.zeros(img.shape)
        lane_visualization(img, h_samples, lanes)

def read_data():
    path = r"D:/TuSimple/train_set/"
    label_data = r"label_data_0531.json"
    load_json_data(path, label_data)

if __name__ == '__main__':
    read_data()