import numpy as np
from matplotlib import pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from label_transform import label_to_int
from label_transform import int_to_string

#Load images with numpy
train_size = 8000
data_train = np.load('X_train_clean_140_45.npy', encoding='latin1')
X_train = data_train[:train_size]
X_valid = data_train[train_size:]

train_labels = np.loadtxt('../input/train_labels.csv', delimiter=',', skiprows=1, dtype='str')
y = np.array([[label_to_int(e[1])] for e in train_labels])
y_train = y[:train_size]
y_valid = y[train_size:]

print(X_train.shape)
print(X_valid.shape)

print(y_train.shape)
print(y_valid.shape)

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(10000,)),
	keras.layers.Dense(2000, activation=tf.nn.relu),
	keras.layers.Dense(2000, activation=tf.nn.relu),
	keras.layers.Dense(31, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

test_loss, test_acc = model.evaluate(X_valid, y_valid)
print('Test accuracy:', test_acc)

In [4]: test_data = np.load('X_test_clean_140_45.npy', encoding='latin1')


with open(fname, 'w') as f:
    ...:     f.write('Id,Category\n')
    ...:     n = 0
    ...:     for x in results:
    ...:         f.write(str(n) + ','+ str(int_to_string[x])+'\n')
    ...:         n+=1

