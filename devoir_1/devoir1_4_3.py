"""
#4.3
1D densities: From the Iris dataset examples, choose a subset corresponding
to one of the classes (of your choice), and one of the characteristic
features, so that we will be in dimension d = 1 and produce a
single graph (using the plot function) including:
(a) the data points of the subset (displayed on the x axis).
(b) a plot of the density estimated by your parametric Gaussian estimator.
(c) a plot of the density estimated by the Parzen estimator with a
hyper-parameter σ (standard deviation) too small.
(d) a plot of the density estimated by the Parzen estimator with the
hyper-parameter σ being a little too big.
(e) a plot of the density estimated by the Parzen estimator with the
hyper-parameter σ that you consider more appropriate. Use a
different color for each plot, and provide your graph with a clear
legend.
(f) Explain how you chose your hyper-parameter σ
"""
import numpy as np
import devoir1_4_1
import devoir1_4_2
import matplotlib.pyplot as plt

data = np.loadtxt('iris.txt')

#take only class 1 and only one features (Sepal Length)
data = data[0:50,0]

#x axis
min_x, max_x = min(data), max(data)
x = np.array(np.linspace(min_x, max_x))

#b)
model_b = devoir1_4_1.Diag_gaussian()
model_b.train(data)
y_b = [model_b.predict(np.array([e])) for e in x]
plt.plot(x, y_b, 'b', label='parametric Gaussian')

#c)
model_c = devoir1_4_2.Kernel_density_estimator(h=0.001)
model_c.train(data)
y_c = [model_c.predict(np.array([e])) for e in x]
plt.plot(x,y_c,'g', label='Parzen (sigma too small)')

#d)
model_d = devoir1_4_2.Kernel_density_estimator(h=0.05)
model_d.train(data)
y_d = [model_d.predict(np.array([e])) for e in x]
plt.plot(x,y_d,'r', label='Parzen (sigma too big)')

#e)
model_e = devoir1_4_2.Kernel_density_estimator(h=0.007)
model_e.train(data)
y_e = [model_e.predict(np.array([e])) for e in x]
plt.plot(x,y_e,'y', label='Parzen (best sigma)')

plt.ylabel('log density')
plt.xlabel('Sepal Length')
plt.legend(loc='best')
plt.show()

#f) expliquer choix de sigma
"""
Le premier sigma est trop petit, car il suit trop les points des données d'entrainement.
Alors, la courbe résultante n'est pas lisse et change de valeur brusquement (exemple: x=5.4 et x=5.6).
Le deuxième sigma est trop grand, car la courbe ne capture pas les subtilités des données et elle ne forme qu'une large cloche.
Le troisième sigma représente un meilleur sigma, car la courbe montre quelques subtilités des données tout en restant lisse
et sans avoir de grosses variances de densité entre des valeurs proches.
"""
