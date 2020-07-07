import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_data():
    data1 = unpickle('cifar/data_batch_1')
    data2 = unpickle('cifar/data_batch_2')
    data3 = unpickle('cifar/data_batch_3')
    data4 = unpickle('cifar/data_batch_4')
    data5 = unpickle('cifar/data_batch_5')

    label_names = unpickle('cifar/batches.meta')

    data = np.concatenate((data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data']))
    labels = np.concatenate((data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels'], data5[b'labels']))
    labels_onehot = pd.get_dummies(labels)

    X_train, X_val, y_train, y_val = train_test_split(data, labels_onehot, test_size=0.2, random_state=20)
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    N1, _ = X_train.shape
    N2, _ = X_val.shape

    X_train = X_train.reshape((N1, 3, 32, 32))
    X_val = X_val.reshape((N2, 3, 32, 32))

    return X_train, X_val, y_train, y_val

    # plt.imshow(x)
    # plt.savefig('a.png')








extract_data()