"""
#4.4
2D densities: Now add a second characteristic feature of Iris, in order
to have entries in d = 2 and produce 4 plots, each displaying the points
of the subset of the data (with the plot function ), and the contour
lines of the density estimated (using the contour function):
(a) by the diagonal Gaussian parametric estimator.
(b) by the Parzen estimator with the hyper-parameter σ (standard
deviation ) being too small.
(c) by the Parzen estimator with the hyper-parameter σ being a little
too big.
(d) by the Parzen estimator with the hyper-parameter σ that you
consider more appropriate.
(e) Explain how you chose your hyper-parameter σ
"""

import numpy as np
import devoir1_4_1
import devoir1_4_2
import matplotlib.pyplot as plt

data = np.loadtxt('iris.txt')
#take only class 1 and two features (Sepal Width and Petal length)
data = data[0:50,1:3]
x = data[:,0]
y = data[:,1]

#a)
model_a = devoir1_4_1.Diag_gaussian()
model_a.train(data)
plt.plot(data[:,0],data[:,1], 'bo', label='parametric Gaussian')

#calcule les densites pour tous les points de la grille
min_x, max_x = min(x), max(x)
min_y, max_y = min(y), max(y)

#want to have a range to see every data point
x_range = np.linspace(min_x -0.1,max_x + 0.1,50)
y_range = np.linspace(min_y - 0.1,max_y + 0.1 ,50)
XY = np.array(np.meshgrid(x_range,y_range))
Z = np.zeros((50,50))

#calculate the densities in the grid
for i in range(XY.shape[1]):
    for j in range(XY.shape[2]):
        Z[i,j] = model_a.predict(np.array([XY[0,i,j], XY[1,i,j]]))
plt.contourf(x_range,y_range,Z)
plt.ylabel('Petal Length')
plt.xlabel('Sepal Width')
plt.legend(loc='best')
plt.show()

"""
#c)
model_c = devoir1_4_2.Kernel_density_estimator(h=0.001)
model_c.train(data)
y_c = [model_c.predict(e) for e in data]
plt.plot(data,y_c,'g', label='Parzen (sigma too small)')

#d)
model_d = devoir1_4_2.Kernel_density_estimator(h=0.01)
model_d.train(data)
y_d = [model_d.predict(e) for e in data]
plt.plot(data,y_d,'r', label='Parzen (sigma too big)')

#e)
model_e = devoir1_4_2.Kernel_density_estimator(h=0.05)
model_e.train(data)
y_e = [model_e.predict(e) for e in data]
plt.plot(data,y_e,'y', label='Parzen (better sigma)')

plt.ylabel('log density')
plt.xlabel('Sepal Width')
plt.legend(loc='best')
plt.show()

#TODO f) expliquer choix de sigma

"""