import numpy as np
import random
random.seed(3)
"""
Transform vector of 10000 examples from train_images and test_images to matrices with (10000,10000) (samples,features) 
"""
images_train = np.load('input/train_images.npy', encoding='latin1')
X = images_train[:,1]
X_train = np.ndarray(shape=(10000, 10000), dtype=float)
for x in range(len(X)):
	for y in range(len(X[0])):
		X_train[x][y] = X[x][y]
X_train.dump("X_train.npy")

images_test = np.load('input/test_images.npy', encoding='latin1')
X = images_test[:,1]
X_test = np.ndarray(shape=(10000, 10000), dtype=float)
for x in range(len(X)):
	for y in range(len(X[0])):
		X_test[x][y] = X[x][y]
X_test.dump("X_test.npy")


#test with less example

# images_train = np.load('input/train_images.npy', encoding='latin1')
# X = images_train[:5,1]
# X_train = np.ndarray(shape=(5, 10000), dtype=float)
# for x in range(len(X)):
# 	for y in range(len(X[0])):
# 		X_train[x][y] = X[x][y]
# X_train.dump("X_train.npy")
#
# images_test = np.load('input/test_images.npy', encoding='latin1')
# X = images_test[:5,1]
# X_test = np.ndarray(shape=(5, 10000), dtype=float)
# for x in range(len(X)):
# 	for y in range(len(X[0])):
# 		X_test[x][y] = X[x][y]
# X_test.dump("X_test.npy")
