import torch
import torch.nn.functional as F
import cv2
import os

def visualize_result(scores, threshold, t, epoch):
    if not os.path.exists('output'):
        os.makedirs('output')
    with torch.no_grad():
        res = torch.zeros((scores.shape[2], scores.shape[3]))
        if scores.shape[1] > 1:  # Multi channel output
            scores = F.softmax(scores, dim=1)
            scores = scores.to('cpu')  # Changing device
            scores = torch.sum(scores, dim=1)
            scores = scores > threshold
            res = scores[0, :, :]
        else:
            scores = F.sigmoid(scores)
            scores = scores.to('cpu')  # Changing device
            scores = scores > threshold
            res = scores[0, 0, :, :]
        res = res.numpy().astype(float)
        # Res has the same dimension as the grayscale image
        res = res * 255.0
        path = os.path.join(os.getcwd(), 'output')
        filename = path + '/' + str(epoch) + '_' + str(t) + '.jpg'
        cv2.imwrite(filename, res)