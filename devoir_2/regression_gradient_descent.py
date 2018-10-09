"""Maxime Daigle & Olivier St-Laurent"""

import numpy as np
import matplotlib.pyplot as plt

"""
1.
implement regression_gradient.
parameters w and b that we want to learn on the training set.
an hyper-parameter to control the capacity of our model: λ. 
There are also hyper-parameters for the optimization: the step-size η, and potentially the number of steps.
"""
class Regression_gradient:

    def __init__(self, capacity, step_size, nb_steps):
        self.capacity = capacity
        self.step_size = step_size
        self.nb_steps = nb_steps

    def train(self,train_data):
        n = train_data.shape[0] #number of example
        d = train_data.shape[1] #number of features (with one added for the bias)
        X = np.column_stack((np.ones(n, dtype=np.float64), train_data[:, 0:d-1])) #add a column of 1 for the bias
        theta = np.random.rand(d) #randomly initiliaze theta with value in [0,1)
        T = train_data[:,-1] #target values
        for i in range(self.nb_steps):
            gradient = 2*np.dot(X.T, np.dot(X,theta) - T)
            #Ridge is the squared norm of the weights (but not the bias). (i.e ||w||^2)
            #ridge_component is the gradient of lambda * ||w||^2
            ridge_component = 2*self.capacity*theta
            ridge_component[0] = 0
            theta = theta - self.step_size*(gradient+ridge_component)
        self.theta = theta

    def predict(self, x):
        X = np.column_stack((np.ones(x.shape[0]),x))
        return np.dot(X,self.theta)

"""
2.
h(x) = sin(x) + 0.3x − 1 
Draw a dataset Dn of pairs (x, h(x)) with n = 15 points where x is drawn uniformly at random in the interval [−5, 5]. 
Make sure to use the same set Dn for all the plots below.
"""
np.random.seed(0)

def h(x):
    return np.sin(x) + 0.3*x - 1

x = np.random.uniform(-5,5,15)

#dataset of 15 pairs (x, h(x))
dataset = np.array((x,h(x)), dtype=np.float64).T


"""
3.
With λ = 0, train your model on Dn with the algorithm regression_gradient.
Then plot on the interval [−10, 10]: the points from the training set Dn, the curve h(x), and the curve of the function 
learned by your model using gradient descent. Make a clean legend. 

Remark: The solution you found with gradient descent should converge to the straight line that is closer from the 
n points (and also to the analytical solution). Be ready to adjust your step-size (small enough) and number
of iterations (large enough) to reach this result.
"""
x_axis = np.array(np.linspace(-10, 10), dtype=np.float64)
plt.plot(dataset[:,0], dataset[:,1], 'bo', label='points from dataset')
hx_curve = h(x_axis)
plt.plot(x_axis,hx_curve,'g', label='curve h(x)')

#curve learned by the model using gradient descent and lambda = 0
model_lambda0 = Regression_gradient(capacity=0, step_size=0.001, nb_steps=500)
model_lambda0.train(dataset)
plt.plot(x_axis, model_lambda0.predict(x_axis), 'r', label='lambda = 0')

"""
4.
on the same graph, add the predictions you get for intermediate value of λ, and for a large value of λ. 
Your plot should include the value of λ in the legend. It should illustrate qualitatively what happens when
λ increases.
"""
#curve with intermediate value of lambda
model_intermediate = Regression_gradient(capacity=30, step_size=0.001, nb_steps=500)
model_intermediate.train(dataset)
plt.plot(x_axis, model_intermediate.predict(x_axis), 'y', label='lambda = 30')

#curve with large value of lambda
model_large = Regression_gradient(capacity=100, step_size=0.001, nb_steps=500)
model_large.train(dataset)
plt.plot(x_axis, model_large.predict(x_axis), 'k', label='lambda = 100')

plt.ylabel('h(x) = sin(x) + 0.3x − 1')
plt.xlabel('x')
plt.legend(loc='best')
plt.show()

"""
5.
Draw another dataset Dtest of 100 points by following the same procedure as Dn. 
Train your linear model on Dn for λ taking values in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]. 
For each value of λ, measure the average quadratic loss on Dtest. 
Report these values on a graph with λ on the x-axis and the loss value on the y-axis.
"""
x_test = np.random.uniform(-5,5,100)
#dataset of 100 pairs (x_test, h(x_test))
dtest = np.array((x_test,h(x_test))).T

capacities = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

#for each capacity, train a model on dataset and calculate his average quedratic loss on dtest
average_losses = []
for e in capacities:
    model = Regression_gradient(capacity=e, step_size=0.001, nb_steps=500)
    model.train(dataset)
    y = model.predict(dtest[:,0])
    losses = np.power(y - dtest[:,1],2)
    average_losses.append(np.mean(losses))

plt.scatter(capacities, average_losses, color = ['red','green','blue','yellow', 'cyan', 'magenta', 'black'])
plt.ylabel('average quadratic loss')
plt.xlabel('lambda')
plt.title('ridge regression with gradient descent')
plt.show()