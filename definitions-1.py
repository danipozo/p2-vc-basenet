from tensorflow.keras.layers import (Conv2D, Conv2DTranspose,
                                     MaxPooling2D, Flatten,
                                     Dense, BatchNormalization,
                                     ReLU, Dropout, Softmax)

from tensorflow import keras

import experiment_utils

'''
Experiment 1.

The goal of this experiment is to test the effect of model depth on accuracy on
CIFAR-100. Also, variations of Basenet with more filters in convolutional layers
will be tested.
'''

# Model 1: Basenet
basenet = keras.Sequential(
    [
        Conv2D(6, kernel_size=5, padding='valid', activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(16, kernel_size=5, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(25),
        Softmax()
    ],
    name='basenet'
)

# Model 2: Basenet, with more filters
model2 = keras.Sequential(
    [
        Conv2D(60, kernel_size=5, padding='valid', activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(60, kernel_size=5, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(25),
        Softmax()
    ],
    name='model2'
)

# Model 3: Basenet, with more layers
model3 = keras.Sequential(
    [
        Conv2D(6, kernel_size=5, padding='valid', activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(16, kernel_size=5, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2DTranspose(16, kernel_size=5, strides=(2,2), padding='valid'),
        Conv2D(16, kernel_size=5, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(25),
        Softmax()
    ],
    name='model3'
)

# Model 4: Basenet, with more layers and more filters
model4 = keras.Sequential(
    [
        Conv2D(60, kernel_size=5, padding='valid', activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(60, kernel_size=5, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2DTranspose(16, kernel_size=5, strides=(2,2), padding='valid'),
        Conv2D(60, kernel_size=5, padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(25),
        Softmax()
    ],
    name='model4'
)

experiment_utils.save_models([basenet, model2, model3, model4],
                             'model-files/def-1')
