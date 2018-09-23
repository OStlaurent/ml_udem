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
#take only class 1 and only one features (Sepal Width)
data = data[0:50,1]
data.sort()

#b)
model_b = devoir1_4_1.Diag_gaussian()
model_b.train(data)
y_b = [model_b.predict(e) for e in data]
plt.plot(data,y_b, 'b', label='parametric Gaussian')

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
print('Scott\'s Rule: n**(-1./(d+4)) =', 50**(-1./(1+4)))
print('Silverman\'s Rule: (n * (d + 2) / 4.)**(-1. / (d + 4))', (50 * (1 + 2) / 4.)**(-1. / (1 + 4)))
print('Bandwidth selection strongly influences the estimate obtained from the KDE (much more so than the actual shape of the kernel). Bandwidth selection can be done by a “rule of thumb”, by cross-validation, by “plug-in methods” or by other means; see [3], [4] for reviews. gaussian_kde uses a rule of thumb, the default is Scott’s Rule.')