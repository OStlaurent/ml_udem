import numpy as np

"""
#4.2
Implement a Parzen density estimator with an isotropic Gaussian kernel.
It will have to work for data of arbitrary dimension d. Likewise it
should have a train() method and a predict() method that computes
the log density
"""
#from tp
def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

def isotropic_gaussian(mu, variance, x, dist_func):
    """
    :param mu: matrix of shape(n,d) containing n training points X_i
    :param x: test point of shape (d,)
    :return: vector for each X_i in train_data/mu (vector shape = (n,))
    """
    return np.exp(-0.5 * np.power(dist_func(mu, x), 2) / variance) / np.power(2 * np.pi * variance, mu.shape[1] / 2)

class kernel_density_estimator:
    def __init__(self, h=1, kernel=isotropic_gaussian, dist_func=minkowski_mat):
        self.h = h
        self.kernel = kernel
        self.dist_func = dist_func

    def train(self,train_data):
        self.train_data = train_data

    def predict(self, x):
        #make sure that shape is correct
        x.shape = (self.train_data.shape[1],)

        #kernel retourne un vecteur de tous les kernel(X_i,x)
        return np.log(np.mean( self.kernel(mu=self.train_data,variance=self.h, x=x, dist_func=self.dist_func)))


if __name__ == '__main__':
    #example
    training_data = np.zeros((100, 3))
    training_data[:, 0] = np.random.normal(0, 0.1, 100)
    training_data[:, 1] = np.random.normal(10, 0.5, 100)
    training_data[:, 2] = np.random.normal(-4, 1, 100)

    model = kernel_density_estimator(h=3)
    model.train(training_data)
    for i in range(100):
        test_data = np.array([np.random.normal(0, 0.1, 1),np.random.normal(10, 0.5, 1), np.random.normal(-4, 1, 1)])
        print(model.predict(test_data))