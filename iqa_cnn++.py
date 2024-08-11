import tensorflow as tf
import numpy as np
import scipy
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from tensorflow.keras import layers, models
 
if os.name =='posix':
    from data_loader_linux import tid2013_loader, kadid10k_loader
else:
    from data_loader import tid2013_loader, kadid10k_loader


model = models.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='linear')
])

data_loader = tid2013_loader()

X_train, y_train = data_loader.train
X_val, y_val = data_loader.val
X_test, y_test = data_loader.test

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=2)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test MAE:', mae)
