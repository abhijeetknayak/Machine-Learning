from first_network import NN
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

model = NN(X_train / 16.0, y_train)

model.load_weights('weights.json')

def get_prediction_vs_GT(m, x, y):
    for xx, yy in zip(x, y):
        s = m.predict(xx)
        print("Prediction : {}, Ground Truth : {}".format(s, np.argmax(yy)))

get_prediction_vs_GT(model, X_val / 16.0, np.array(y_val))