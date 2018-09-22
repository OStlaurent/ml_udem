import numpy as np

"""
#4.2
Implement a Parzen density estimator with an isotropic Gaussian kernel.
It will have to work for data of arbitrary dimension d. Likewise it
should have a train() method and a predict() method that computes
the log density
"""

def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

#TODO
def isotropic_gaussian(mu,variance,x):
    pass

class kernel_density_estimator:
    def __init__(self, h=1, kernel=isotropic_gaussian(), dist_func=minkowski_mat):
        self.h = h
        self.dist_func = dist_func
        self.kernel = kernel

    #TODO train fait autre chose?
    def train(self,train_data):
        self.train_data = train_data

    #TODO
    def predict(self, x):
        #TODO probleme
        #on veut means i=i to n of Kernel(X_i,x) mais la
        #on a means of kernel(X,x) ou X est la matrice de toutes les points X_i
        #alors il faudrait que isotropic_gaussian retourne un vecteur de tous les kernerl(X_i,x)
        #np.means( self.kernel(mu=self.train_data,variance=self.h, x=x))
        pass


if __name__ == '__main__':
    #example
    training_data = np.zeros((100, 3))
    training_data[:, 0] = np.random.normal(0, 0.1, 100)
    training_data[:, 1] = np.random.normal(10, 0.5, 100)
    training_data[:, 2] = np.random.normal(-4, 1, 100)

    model = kernel_density_estimator()
    model.train(training_data)
    for i in range(100):
        test_data = np.array([np.random.normal(0, 0.1, 1),np.random.normal(10, 0.5, 1), np.random.normal(-4, 1, 1)])
        print(model.predict(test_data))