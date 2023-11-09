from tensorflow.keras import datasets, layers, models
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='fine')
train_images, test_images = train_images / 255.0, test_images / 255.0
model_100 = models.Sequential()
model_100.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_100.add(layers.MaxPooling2D((2, 2)))
model_100.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_100.add(layers.MaxPooling2D((2, 2)))
model_100.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_100.add(layers.Flatten())
model_100.add(layers.Dense(64, activation='relu'))
model_100.add(layers.Dense(100)) 
model_100.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_100 = model_100.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode='coarse')
train_images, test_images = train_images / 255.0, test_images / 255.0
model_20 = models.Sequential()
model_20.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_20.add(layers.MaxPooling2D((2, 2)))
model_20.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_20.add(layers.MaxPooling2D((2, 2)))
model_20.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_20.add(layers.Flatten())
model_20.add(layers.Dense(64, activation='relu'))
model_20.add(layers.Dense(20))
model_20.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history_20 = model_20.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
model_100.save('100.keras')
model_20.save('20.keras')