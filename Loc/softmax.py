import numpy as np
from data_utils import load_data

class SoftmaxRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

if __name__ == '__main__':
    path = 'data/'
    x_train, y_train = load_data(path)
    print(x_train.shape, y_train.shape)