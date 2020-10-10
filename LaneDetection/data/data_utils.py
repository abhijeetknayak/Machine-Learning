import numpy as np
import json
import cv2
import os
from easydict import EasyDict as edict
from LaneDetection.config import *
import torch
from torch.utils.data import Dataset
import glob

class TuSimpleData(Dataset):
    def __init__(self, path=None, data_list='train'):
        self.image_list = []
        self.gt = []
        self.data_path = path

        self.load_json_data()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.data_path + self.image_list[idx])
        mask = np.zeros(image.shape)
        for lane in self.gt[idx]:
            cv2.polylines(mask, np.int32([lane]), isClosed=False, color=(255, 255, 255), thickness=5)

        return image, mask

    def load_json_data(self):
        for file in os.listdir(self.data_path):
            if file.endswith('.json'):
                with open(self.data_path + file) as json_file:
                    for line in json_file:
                        data = edict(json.loads(line))
                        self.image_list.append(data.raw_file)
                        h_samples = data.h_samples
                        lanes = data.lanes
                        self.gt.append(self.ground_truth(h_samples, lanes))

    def ground_truth(self, h_samples, lanes):
        lane_viz = [[(x, y) for (x, y) in zip(lane, h_samples) if x >= 0] for lane in lanes]
        return lane_viz


def lane_visualization(img, h_samples, lanes):
    lane_viz = [[(x, y) for (x, y) in zip(lane, h_samples) if x >= 0] for lane in lanes]
    for lane in lane_viz:
        cv2.polylines(img, np.int32([lane]), isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    data = TuSimpleData(path=r"D:/TuSimple/train_set/")

    img, label = data.__getitem__(10)
    cv2.imshow("Image", img)
    cv2.imshow("Mask", label)
    cv2.waitKey(0)