# DNN service to identify numbers from handwritten pictures of numbers
from __future__ import absolute_import, division, print_function, unicode_literals

import keras.utils.np_utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # now import the tensorflow module
# Get pre-cooked data included in keras library for ease of use
mnist = tf.keras.datasets.mnist  # 28 x 28 images of handwritten digits 0-9
# Normalize the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = keras.utils.np_utils.normalize(x_train, axis=1)
x_test = keras.utils.np_utils.normalize(x_test, axis=1)
# Create a model (reference static/images/deepNeuralNetwork.png)
model = keras.models.Sequential()
# Flatten the tensor for better results in the model
model.add(tf.keras.layers.Flatten())
# Typical hidden layer 2
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Typical hidden layer 2
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# We need a neuron per output and the softmax activation function to pick the higher weight in the last layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
# Define parameters for the training of the model
model.compile(optimizer='adam',  # `Adam` is a default optimizer, you could also use `stochastic gradient descent`
              loss='sparse_categorical_crossentropy',  # `*_categorical_crossentropy` is the go-to method to calculate loss, it can be used when we have 2 or more categories, it yields the same results as binary, hence it's better to use it as default
              metrics=['accuracy'])
# Train the model with 3 repetitions
model.fit(x_train, y_train, epochs=3)
# Calculate validation loss anb validation accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
# Observe you don't have a huge delta, that means the model was over-fitted
print(val_loss, val_acc)
print(x_train[0])
# Save the model
model.save('dnn_num_reader.model')
# Load the model
new_model = tf.keras.models.load_model('dnn_num_reader.model')
# Make a prediction
predictions = new_model.predict([x_test])  # Always takes a list
# Friendly print the predictions
print(np.argmax(predictions[0]))  # It's a seven!
# Friendly print the input to double check
plt.imshow(x_test[0])
plt.show()
print('Skynet has become self-aware')
