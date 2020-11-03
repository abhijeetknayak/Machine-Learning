import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from LaneDetection.config import config
from LaneDetection.data.data_utils import TuSimpleData
from LaneDetection.model.erfnet import Erfnet
from LaneDetection.visualize import *
import cv2
import os

def initialize_weights(model, type=''):
    pass

def define_optimizer(params, weight_decay=0, type='sgd', lr=1e-3):
    if type == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif type == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    return optimizer

def soft_dice_loss(scores, labels):
    union = torch.sum(torch.mul(scores, labels))
    union = torch.mul(2, union)
    scores_sum_sqr = torch.sum(torch.square(scores))
    labels_sum_sqr = torch.sum(torch.square(labels))

    loss = torch.sub(1, torch.div(union, torch.add(scores_sum_sqr, labels_sum_sqr)))

    return loss

def mseloss(scores, labels):
    loss = torch.mean(torch.square(torch.sub(scores, labels)))

    return loss


def save_checkpoint(state, iteration, filename='checkpoint.pth'):
    if not os.path.exists('trained'):
        os.makedirs('trained')
    filename = 'trained/' + str(iteration) + filename
    torch.save(state, filename)

def train(model, train_path, max_epochs=100, print_every=10, save_every=100, save_image=50):
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1}
    dtype = torch.float32

    training_set = TuSimpleData(path=train_path, num_classes=5)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    for epoch in range(max_epochs):  # Number of epochs
        for t, (images, labels) in enumerate(training_generator):
            images, labels = images.to(device, dtype=dtype), labels.to(device, dtype=dtype)
            model.train()  # Train mode

            scores = model(images)
            loss = soft_dice_loss(scores, labels)
            # loss = mseloss(scores, labels)
            # loss = torch.add(loss1, loss2)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if t % save_image == 0:
                visualize_result(scores, 0.5, t, epoch)

            if t % print_every == 0:
                print("Epoch : {}; Iteration {}; Loss : {}".format(epoch, t, loss))

            if t % save_every == 0:
                save_checkpoint({'epoch': epoch + 1,
                                 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'loss': loss
                                 }, t)

def binary_segmentation_train(
        model, train_path, max_epochs=100, print_every=10, save_every=100, save_image=50):
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 1}
    dtype = torch.float32

    training_set = TuSimpleData(path=train_path, num_classes=1)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    for epoch in range(max_epochs):  # Number of epochs
        for t, (images, labels) in enumerate(training_generator):
            images, labels = images.to(device, dtype=dtype), labels.to(device, dtype=dtype)
            model.train()  # Train mode

            scores = model(images)
            weights = torch.tensor([0.1, 0.98])
            loss = F.binary_cross_entropy_with_logits(scores[:, 0, :, :], labels, weight=weights)
            # loss = soft_dice_loss(scores, labels)
            # loss = mseloss(scores, labels)
            # loss = torch.add(loss1, loss2)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if t % save_image == 0:
                visualize_result(scores, 0.5, t, epoch)

            if t % print_every == 0:
                print("Epoch : {}; Iteration {}; Loss : {}".format(epoch, t, loss))

            if t % save_every == 0:
                save_checkpoint({'epoch': epoch + 1,
                                 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'loss': loss
                                 }, t, 'binary_seg_ckpt.pth')

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    multi_class = 0
    train_path = r"C:/Users/NAYAKAB/Desktop/dataset/"
    # train_path = r"D:/TuSimple/train_set/"


    if multi_class:
        checkpoint = torch.load('trained/0checkpoint.pth')

        model = Erfnet(5)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = define_optimizer(model.parameters(), type='adam', lr=1e-3, weight_decay=0.001)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("Starting from epoch {}".format(checkpoint['epoch']))

        model.to(device)

        train(model, train_path, max_epochs=10000, save_image=500)
    else:
        model = Erfnet(1)
        optimizer = define_optimizer(model.parameters(), type='adam', lr=1e-3)

        model.to(device)

        binary_segmentation_train(model, train_path, max_epochs=10000)



