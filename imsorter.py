import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os

labels = ['smug/', 'notsmug/']
size = 400, 400

# Function for loading dataset
def get_images(location):
    data = []
    for class_n, label in enumerate(labels):
        im_path = os.path.join(location, label)
        for x in os.listdir(im_path):
            im = Image.open(im_path + x)
            im.thumbnail(size, Image.ANTIALIAS)
            im_arr = np.asarray(im)
            data.append([im_arr, class_n])
        return np.array(data, dtype=object)

train = get_images('/home/pon-k/PycharmProjects/pythonProject/venv/mlsample/train/')
test = get_images('/home/pon-k/PycharmProjects/pythonProject/venv/mlsample/test/')

# Prepare the training models
x_train = []
y_train = []
x_test = []
y_test = []

for ftr, label in train:
    x_train.append(ftr)
    y_train.append(label)

for ftr, label in test:
    x_test.append(ftr)
    y_test.append(label)

x_train = np.array(x_train, dtype=object) / 255
x_test = np.array(x_test, dtype=object) / 255

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)