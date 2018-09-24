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
data_x = data[:,0]
data_y = data[:,1]

# create grid of values where we want the densities
min_x, max_x = min(data_x), max(data_x)
min_y, max_y = min(data_y), max(data_y)
x = np.linspace(min_x -0.1,max_x + 0.1,50)
y = np.linspace(min_y - 0.1,max_y + 0.1 ,50)
XY = np.array(np.meshgrid(x,y))

def grid_densities(XY, model):
    """
    :param XY: grid of values where we want the densities
    :param model: model used to calculate the densities
    :return: densities on the grid
    """
    Z = np.zeros((50, 50))
    for i in range(XY.shape[1]):
        for j in range(XY.shape[2]):
            Z[i, j] = model.predict(np.array([XY[0, i, j], XY[1, i, j]]))
    return Z

def make_and_show_plot(x,y,Z):
    """
    :param x: vector length n
    :param y: vector length m
    :param Z: matrix nxm
    make and show the ocntour plot
    """
    plt.contourf(x, y, Z)
    plt.ylabel('Petal Length')
    plt.xlabel('Sepal Width')
    plt.legend(loc='upper left')
    plt.show()

#a)
model_a = devoir1_4_1.Diag_gaussian()
model_a.train(data)
plt.plot(data_x,data_y, 'bo', label='parametric Gaussian')
Z_a = grid_densities(XY,model_a)
make_and_show_plot(x,y,Z_a)

#b)
model_b = devoir1_4_2.Kernel_density_estimator(h=0.001)
model_b.train(data)
plt.plot(data_x,data_y, 'bo', label='Parzen (sigma too small)')
Z_b = grid_densities(XY,model_b)
make_and_show_plot(x,y,Z_b)

#c)
model_c = devoir1_4_2.Kernel_density_estimator(h=1)
model_c.train(data)
plt.plot(data_x,data_y, 'bo', label='Parzen (sigma too big)')
Z_c = grid_densities(XY,model_c)
make_and_show_plot(x,y,Z_c)

#d)
model_d = devoir1_4_2.Kernel_density_estimator(h=0.04)
model_d.train(data)
plt.plot(data_x,data_y, 'bo', label='Parzen (best sigma)')
Z_d = grid_densities(XY,model_d)
make_and_show_plot(x,y,Z_d)

#e)
"""
Le premier sigma est trop petit, car le modèle apprend seulement les points des données d'entrainement et ne généralise pas.
Alors, le modèle encercle chacune des données d'entrainement.
Le deuxième sigma est trop grand, car le modèle ne fait que des cercles réguliers centrés sur la moyenne. Il ne capture pas
de subtilités.
Le troisième sigma est meilleur, car le modèle apprend à suivre les points sans avoir juste mémoriser les points. 
Les cercles représentent bien une généralisation de la distribution des points d'entrainement.
"""