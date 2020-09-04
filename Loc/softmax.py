import numpy as np
from data_utils import *

class SoftmaxRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

if __name__ == '__main__':
    path = 'data/'
    # x_train, y_train = load_data(path)
    x_train, y_train = load_data_numpy(path)
    # print(x_train.shape, y_train.shape)
    x_train = select_features(x_train)