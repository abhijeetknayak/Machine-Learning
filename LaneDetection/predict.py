import torch
import cv2
from LaneDetection.model.erfnet import Erfnet
import numpy as np
import torch.nn.functional as F

def read_checkpoint(image_path):
    model = Erfnet(5)
    device = 'cuda:0'

    model.load_state_dict(torch.load('trained/800checkpoint.pth')['state_dict'])
    model.to(device)

    img = cv2.imread(image_path)

    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :, :, :]

    img = torch.from_numpy(img)

    img = img.to(device, dtype=torch.float32)
    with torch.no_grad():
        scores = model(img)

        print(F.softmax(scores, dim=1).shape)
        scores = scores > 0.5
        scores = scores.to('cpu').numpy()
        print(scores[0, 0, :, :].astype(float))
        cv2.imshow("im", scores[0, 0, :, :].astype(float))
        cv2.waitKey(0)

if __name__ == '__main__':
    read_checkpoint(r'D:\TuSimple\test_set\clips\0601\1494453529590660995\4.jpg')