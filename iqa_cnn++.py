import tensorflow as tf
import numpy as np
import scipy
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from data_loader import tid2013_loader
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Dropout, Input, Flatten



def create_data_generators(base_dir, train_data, val_data, test_data, batch_size):
    datagen = ImageDataGenerator()
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=os.path.join(base_dir, 'training/patches/'),
        x_col='image',
        y_col=['MOS', 'distortion_encoded'],
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='multi_output',
        shuffle=True,
        seed=42
    )
    val_generator = datagen.flow_from_dataframe(
        dataframe=val_data,
        directory=os.path.join(base_dir, 'validation/patches/'),
        x_col='image',
        y_col=['MOS', 'distortion_encoded'],
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='multi_output',
        shuffle=True,
        seed=42
    )
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_data,
        directory=os.path.join(base_dir, 'test/patches/'),
        x_col='image',
        y_col=['MOS', 'distortion_encoded'],
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='multi_output',
        shuffle=False,
        seed=42
    )
    return train_generator, val_generator, test_generator

def define_model(input_shape = (32, 32, 3)):
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    mos_output = tf.keras.layers.Dense(1, name='dmos')(x)
    distortion_output = tf.keras.layers.Dense(13, activation='softmax', name='distortion')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=[mos_output, distortion_output])
    return model

tid2013_loader = tid2013_loader(download=False)
train, val, test = tid2013_loader.train, tid2013_loader.val, tid2013_loader.test
base_dir = os.path.join(os.getcwd(), 'databases/tid2013/normalized_distorted_images')
train_generator, val_generator, test_generator = create_data_generators(base_dir, train, val, test, batch_size=32)


model = define_model()
model.compile(optimizer='adam',
              loss={'dmos': 'mae', 'distortion': 'categorical_crossentropy'},
              metrics={'dmos': 'mae', 'distortion': 'accuracy'})

history = model.fit(train_generator, epochs=25, validation_data=val_generator, verbose=2)