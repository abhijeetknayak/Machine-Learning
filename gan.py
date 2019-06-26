import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2

from keras_adversarial import simple_gan, AdversarialModel, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

seed = 128
rng = np.random.RandomState(seed)

root_folder = '.'
data_folder = './data/'

train_csv = pd.read_csv('./data/train.csv')

temp = []

for img_name in train_csv.filename:
    img = imread('./data/Images/train/' + img_name, flatten=True).astype('float32')
    temp.append(img)
train_x = np.stack(temp)

train_x /= 255

g_input_shape = 100
d_input_shape = (28, 28)
hidden_1_num_units = 500
hidden_2_num_units = 500
g_output_num_units = 784
d_output_num_units = 1
epochs = 25
batch_size = 128

model_1 = Sequential([Dense(units=hidden_1_num_units, input_dim=g_input_shape,
                                activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
                      Dense(units=hidden_2_num_units,
                            activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
                      Dense(units=g_output_num_units,
                            activation='sigmoid', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
                      Reshape(d_input_shape)])
model_2 = Sequential([InputLayer(input_shape=d_input_shape),
                      Flatten(),
                      Dense(units=hidden_1_num_units, activation='relu',
                            kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
                      Dense(units=hidden_2_num_units, activation='relu',
                            kernel_regularizer=L1L2(l1=1e-5, l2=1e-5)),
                      Dense(units=d_output_num_units, activation='sigmoid',
                            kernel_regularizer=L1L2(l1=1e-5, l2=1e-5))])

model_1.summary()
model_2.summary()

ganModel = simple_gan(model_1, model_2, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=ganModel, player_params=[model_1.trainable_weights, model_2.trainable_weights])

model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                          player_optimizers=['adam', 'adam'], loss='binary_crossentropy')
history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=25, batch_size=batch_size)

# plt.plot(history.history['player_0_loss'])
# plt.plot(history.history['player_1_loss'])
# plt.plot(history.history['loss'])
# plt.show()

zsamples = np.random.normal(size=(10, 100))
pred = model_1.predict(zsamples)
for i in range(pred.shape[0]):
    plt.imshow(pred[i, :], cmap='gray')
    plt.show()