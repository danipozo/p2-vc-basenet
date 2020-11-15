import experiment_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar100
import numpy as np
import tensorflow.keras.utils as utils

import sys
import os
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

'''
Experiment setup 1.

Load models and test them against 25 classes of CIFAR-100.
'''

models_path = sys.argv[1]
results_path = Path(sys.argv[2])

EPOCHS = 60
if os.getenv('EPOCHS') is not None:
    EPOCHS = int(os.getenv('EPOCHS'))

def cargarImagenes():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data (label_mode ='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape (train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]
    
    y_train = utils.to_categorical(y_train, 25)
    y_test = utils.to_categorical(y_test, 25)
    
    return x_train , y_train , x_test , y_test

logging.info('Reading data')
x_train, y_train, x_test, y_test = cargarImagenes()
logging.info('Data read')

# Split training set in training/validation
val_split = 0.1
n_val = int(x_train.shape[0]*val_split)

x_val, y_val = x_train[:n_val], y_train[:n_val]
x_train, y_train = x_train[n_val:], y_train[n_val:]

# Load models
models = experiment_utils.load_models(models_path)

# Declare optimizer
optimizer = Adam()

# Execute experiment
hists = []
for m in models:
    logging.info('Fitting model')
    m.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics='accuracy')
    hist = m.fit(x_train,
                 y_train,
                 batch_size=128,
                 epochs=EPOCHS,
                 validation_data=(x_val,y_val))
    hists.append((m,hist))

logging.info('All models fitted')
    
# Save results
experiment_utils.save_results(hists, results_path)
