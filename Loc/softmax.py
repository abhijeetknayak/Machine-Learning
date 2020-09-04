import numpy as np
from data_utils import *

class SoftmaxRegression:
    def __init__(self, lr=0.01, epochs=10, batch_size=250):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None

    def fit(self, X, y):
        N, D = X.shape
        if self.W is None:
            C = np.max(y) + 1
            self.W = np.zeros((D, C))
            self.b = np.zeros(C)

        for epoch in range(self.epochs):
            iteration = N // self.batch_size
            for it in iteration:
                id = np.random.choice(N, self.batch_size, replace=False)
                x_b, y_b = X[id], y[id]
                scores = x_b.dot(self.W) + self.b
                exp_scores = np.exp(scores)
                prob = exp_scores / np.sum(exp_scores)

    def predict(self, X):
        pass

if __name__ == '__main__':
    path = 'data/'
    out = load_data_df(path)
    print(out['Id'].unique())
    print(out.groupby('Id').count())
    x_train, y_train = load_data_numpy(path)
    x_train = x_train[y_train != 1]
    print(x_train.shape)
    # print(x_train.shape, y_train.shape)
    k = [0, 1, 3, 14, 15, 16, 17, 18, 19]
    x_train = select_features(x_train, k)
    # print(x_train.shape)

    model = SoftmaxRegression()
