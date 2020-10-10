import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from LaneDetection.config import config
from LaneDetection.data.data_utils import TuSimpleData
from LaneDetection.model.erfnet import Erfnet
import cv2

def initialize_weights(model, type=''):
    pass

def define_optimizer(params, weight_decay, type='sgd', lr=1e-3):
    if type == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif type == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    return optimizer


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 6}

    max_epochs = 100
    print_every = 100
    dtype = torch.float32
    training_set = TuSimpleData(path=r"D:/TuSimple/train_set/")
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    model = Erfnet(5)
    # initialize_weights(model, 'kaiming')

    optimizer = define_optimizer(model.parameters(), type='adam', lr=1e-3, weight_decay=0.001)

    model.to(device)

    for epoch in range(max_epochs):  # Number of epochs
        for t, (images, labels) in enumerate(training_generator):
            images, labels = images.to(device, dtype=dtype), labels.to(device)
            model.train()  # Train mode

            scores = model(images)
            print(scores.shape)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print("Iteration {}; Loss : {}".format(t, loss))


