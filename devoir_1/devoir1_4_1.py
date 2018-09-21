import numpy as np

"""
#4.1
Implement a diagonal Gaussian parametric density estimator. It will
have to work for data of arbitrary dimension d. It
should have a train() method to learn the parameters and a method
predict() which calculates the log density.
"""

class diag_gaussian:
    def __init__(self):
        pass

    # The train function compute mu and the covariance matrix
    def train(self,train_data):
        self.train_data = train_data

        #means of each features
        self.mu = np.mean(self.train_data, axis=0)

        #diagonal Gaussian => covariance matrix = diagonal(sigma_1^2, ..., sigma_d^2)
        #sigma_i^2 = (1/n) * sum t=1 to n of (x_t,i - mu_i)^2
        variances = np.mean(np.power(self.train_data - self.mu, 2), axis=0)
        self.cov_matrix = np.diag(variances)

    def predict(self, test_data):
        #make sure that test_data is a column vector
        test_data.shape = self.mu.shape

        # diagonal matrix => (element_i,i of inverse = 1) / (element_i,i of not inverse)
        # inverse_cov_matrix = np.diag(1 / variances)
        # u = test_data - self.mu
        # A = [u_1,u_2,...,u_d] * inverse_cov_matrix = [ u_1 / sigma_1^2, ... , u_d / sigma_d^2]
        # A * [u_1,u_2,...,u_d].transpose() = u_1^2 / sigma_1^2 + ... + u_d^2 / sigma_d^2

        variances = np.diagonal(self.cov_matrix)
        u = test_data - self.mu

        # diagonal matrix => det(matrix) = product i=1 to n of element_i,i
        sqrt_det = np.sqrt(np.prod(variances))

        #log(density)
        return np.log(np.exp(-0.5 * sum(np.power(u,2)/variances)) / (np.power(2*np.pi, test_data.shape[0]/2) * sqrt_det))



if __name__ == '__main__':
    #example
    training_data = np.zeros((100, 3))
    training_data[:, 0] = np.random.normal(0, 0.1, 100)
    training_data[:, 1] = np.random.normal(10, 0.5, 100)
    training_data[:, 2] = np.random.normal(-4, 1, 100)

    model = diag_gaussian()
    model.train(training_data)
    for i in range(100):
        test_data = np.array([np.random.normal(0, 0.1, 1),np.random.normal(10, 0.5, 1), np.random.normal(-4, 1, 1)])
        print(model.predict(test_data))