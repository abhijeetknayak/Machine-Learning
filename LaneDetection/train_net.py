import numpy as np
import torch
from torch.utils.data import DataLoader
from LaneDetection.config import config
from LaneDetection.data.data_utils import TuSimpleData
from LaneDetection.model.erfnet import Erfnet
import cv2

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 3,
              'shuffle': True,
              'num_workers': 6}

    max_epochs = 100
    training_set = TuSimpleData(path=r"D:/TuSimple/train_set/")
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    model = Erfnet(5)
    model.apply(fn='')

    for epoch in range(max_epochs):  # Number of epochs
        for images, labels in training_generator:
            images, labels = images.to(device), labels.to(device)
            model
