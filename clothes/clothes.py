# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:44:47 2024

@author: Arturo
"""
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

len(train_images)

set(list(train_labels))

plt.figure()
plt.imshow(train_images[2])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

class CNN(Model):
  def __init__(self, input_shape, learning_rate):
    super(CNN, self).__init__()
    self.flatten      = Flatten(input_shape=input_shape)
    self.dense        = Dense(512, activation='relu')
    self.dense_1      = Dense(256, activation='relu')
    self.batch_norm   = BatchNormalization()
    self.dropout      = Dropout(0.2)
    self.dense_2      = Dense(128, activation='relu')
    self.batch_norm_1 = BatchNormalization()
    self.dropout_1    = Dropout(0.2)
    self.out_layer    = Dense(10, activation='softmax')

    self.optimizer = Adam(learning_rate=learning_rate)
    self.compile(optimizer = self.optimizer,
                 loss      = 'sparse_categorical_crossentropy',
                 metrics   = ['accuracy']
              )

  def call(self, inputs):
    x = self.flatten(inputs)
    x = self.dense(x)
    x = self.dense_1(x)
    x = self.batch_norm(x)
    x = self.dropout(x)
    x = self.dense_2(x)
    x = self.batch_norm_1(x)
    x = self.dropout_1(x)
    return self.out_layer(x)

model = CNN(input_shape=(28, 28), learning_rate=0.001)
model.fit(train_images, 
          train_labels, 
          epochs=50,
          validation_split=0.2,
          callbacks  = [
                # Create register in TensorBoard
                tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    
                # Save the best model in path
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='./checkpoints/best_model.keras',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                ),
    
                # Stop training model if not improbe in specific case
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=1
                ),
    
                # Save logs of training
                CSVLogger('training_log.csv', separator=',', append=False)
            ]
          )

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

