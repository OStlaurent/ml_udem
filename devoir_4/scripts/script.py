import numpy as np
from matplotlib import pyplot as plt

import os
# print(os.listdir("../input"))

#Load images with numpy
images_train = np.load('../input/train_images.npy', encoding='latin1')
images_train.shape

#Load labels
train_labels = np.genfromtxt('../input/train_labels.csv', names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S15')])
train_labels.shape

X = images_train[:,1]

X_tmp = np.ndarray(shape=(10000, 10000), dtype=float)

for x in range(len(X)):
	for y in range(len(X[0])):
		X_tmp[x][y] = X[x][y]

print(X_tmp[0])
print(type(X_tmp[0]))
print(X_tmp.shape)
print(X_tmp.shape[0])

# print(len(X))
# print(X[0])
# print(type(X[0][0]))
# print(type(X.shape))
# for x in range(5):
# 	print(X[x])

#Reshaping image to 100x100
# for x in range(10):
# 	image_train1 = (X[x]).reshape(100,100)
# 	plt.imshow(image_train1)
# 	plt.show()

# 	#Printing label
# 	print(train_labels[x])

#Load images with numpy
# images_test = np.load('../input/test_images.npy', encoding='latin1')
# images_test.shape

#Reshaping image to 100x100
# image_test1 = (images_test[0][1]).reshape(100,100)
# plt.imshow(image_test1)
# plt.show()