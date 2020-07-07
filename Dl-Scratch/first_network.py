import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import json

digits = load_digits()

target = pd.get_dummies(digits.target)
print(target.shape)

X_train, X_val, y_train, y_val = train_test_split(digits.data, target, test_size=0.2, random_state=20)

# def sigmoid(x):  # Array Input accepted
#     return 1 / (1 + np.exp(-x))
#
# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)
#
# def sigmoid_deriv(x):
#     return np.exp(-x) / np.square(1 + np.exp(-x))

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_deriv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    samples = real.shape[0]
    diff = pred - real
    return diff / samples

def error(pred, real):
    samples = real.shape[0]
    logp = - np.log(pred[np.arange(samples), real.argmax(axis=1)])
    loss = np.sum(logp)/samples
    return loss

class NN:
    def __init__(self, X, y):
        self.input = X
        self.target = y
        self.W1 = np.random.randn(64, 128)
        self.W2 = np.random.randn(128, 128)
        self.W3 = np.random.randn(128, 10)
        self.b1 = np.zeros((1, 128))
        self.b2 = np.zeros((1, 128))
        self.b3 = np.zeros((1, 10))
        self.H1 = None
        self.H2 = None
        self.H3 = None
        self.H1A = None
        self.H2A = None
        self.H3A = None
        self.dW1 = None
        self.dW2 = None
        self.dW3 = None
        self.db1 = None
        self.db2 = None
        self.db3 = None
        self.lr = 0.5

    def feedForward(self):
        H1 = np.dot(self.input, self.W1) + self.b1
        self.H1A = sigmoid(H1)

        H2 = np.dot(self.H1A, self.W2) + self.b2
        self.H2A = sigmoid(H2)

        H3 = np.dot(self.H2A, self.W3) + self.b3
        self.H3A = softmax(H3)

    def backprop(self):
        loss = error(self.H3A, self.target)
        print("Loss : {}".format(loss))
        H3_delta = cross_entropy(self.H3A, self.target)  # (X, 10)

        Z2_Delta = np.dot(H3_delta, self.W3.T)  # (X, 10) * (10, 128) = (X, 128)
        H2_Delta = Z2_Delta * sigmoid_deriv(self.H2A)  # (X, 128). This is an element wise product

        Z1_Delta = np.dot(H2_Delta, self.W2)  # (X, 128) * (128, 128) = (X, 128)
        H1_Delta = Z1_Delta * sigmoid_deriv(self.H1A)  # (X, 128). This is an element wise product.

        self.db1 = np.sum(H1_Delta, axis=0)
        self.dW1 = np.dot(self.input.T, H1_Delta)  # (64, X) * (X, 128) = (64 , 128)
        self.db2 = np.sum(H2_Delta, axis=0)
        self.dW2 = np.dot(self.H1A.T, H2_Delta)  # (128, X) * (X, 128) = (128, 128)
        self.db3 = np.sum(H3_delta, axis=0)  # (128, 10)
        self.dW3 = np.dot(self.H2A.T, H3_delta)  # (128, X) * (X, 10) = (128, 10)

        self.W1 -= self.lr * self.dW1
        self.W2 -= self.lr * self.dW2
        self.W3 -= self.lr * self.dW3
        self.b1 -= self.lr * self.db1
        self.b2 -= self.lr * self.db2
        self.b3 -= self.lr * self.db3

    def predict(self, data):
        self.input = data
        self.feedForward()
        return self.H3A.argmax()

    def dump_weights(self, file):
        json_file = dict()
        json_file['W1'] = list(self.W1.reshape(-1))
        json_file['W2'] = list(self.W2.reshape(-1))
        json_file['W3'] = list(self.W3.reshape(-1))
        json_file['b1'] = list(self.b1.reshape(-1))
        json_file['b2'] = list(self.b2.reshape(-1))
        json_file['b3'] = list(self.b3.reshape(-1))

        with open(file, 'w+') as f:
            json.dump(json_file, f)

    def load_weights(self, file):
        f = open(file)
        json_data = json.load(f)
        try:
            self.W1 = np.array(json_data['W1']).reshape((self.W1.shape[0], self.W1.shape[1]))
            self.W2 = np.array(json_data['W2']).reshape((self.W2.shape[0], self.W2.shape[1]))
            self.W3 = np.array(json_data['W3']).reshape((self.W3.shape[0], self.W3.shape[1]))
            self.b1 = np.array(json_data['b1']).reshape((self.b1.shape[0], self.b1.shape[1]))
            self.b2 = np.array(json_data['b2']).reshape((self.b2.shape[0], self.b2.shape[1]))
            self.b3 = np.array(json_data['b3']).reshape((self.b3.shape[0], self.b3.shape[1]))
        except:
            print("Error in loading weights")

if __name__ == '__main__':
    model = NN(X_train / 16.0, np.array(y_train))

    epochs = 1500
    for x in range(epochs):
        model.feedForward()
        model.backprop()

    model.dump_weights('weights.json')

    model.load_weights('weights.json')



    def get_acc(x, y):
        acc = 0
        for xx, yy in zip(x, y):
            s = model.predict(xx)
            if s == np.argmax(yy):
                acc += 1
        return acc / len(x) * 100


    print("Training accuracy : ", get_acc(X_train / 16, np.array(y_train)))
    print("Test accuracy : ", get_acc(X_val / 16, np.array(y_val)))


