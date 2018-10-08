"""Maxime Daigle & Olivier St-Laurent"""

import numpy as np
import matplotlib.pyplot as plt

"""
1.
implement regression_gradient. Note that we now
have parameters w and b we want to learn on the training set, as well
an hyper-parameter to control the capacity of our model: λ. There
are also hyper-parameters for the optimization: the step-size η, and
potentially the number of steps.
"""
class Regression_gradient:

    def __init__(self, capacity, step_size, nb_steps):
        self.capacity = capacity
        self.step_size = step_size
        self.nb_steps = nb_steps

    def train(self,train_data):
        pass
        #TODO

    def predict(self, x):
        pass
        #TODO

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
dataset = np.array((x,h(x))).T

"""
3.
With λ = 0, train your model on Dn with the algorithm regression_gradient.
Then plot on the interval [−10, 10]: the points from the training set Dn, the curve h(x), and the curve of the function 
learned by your model using gradient descent. Make a clean legend. 

Remark: The solution you found with gradient descent should converge to the straight line that is closer from the 
n points (and also to the analytical solution). Be ready to adjust your step-size (small enough) and number
of iterations (large enough) to reach this result.
"""
x_axis = np.array(np.linspace(-10, 10))
plt.plot(dataset[:,0], dataset[:,1], 'bo', label='points from dataset')
hx_curve = h(x_axis)
plt.plot(x_axis,hx_curve,'g', label='curve h(x)')
#TODO curve learned by the model using gradient descent.



"""
4.
on the same graph, add the predictions you get for intermediate value of λ, and for a large value of λ. 
Your plot should include the value of λ in the legend. It should illustrate qualitatively what happens when
λ increases.
"""
#TODO add curves with intermediate value of lambda and large value of lambda


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
# x_test = np.random.uniform(-5,5,100)
# #dataset of 100 pairs (x_test, h(x_test))
# dtest = np.array((x_test,h(x_test))).T
#
# capacities = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
#
# #TODO for each capacity, train a model on dataset and calculate his average quedratic loss on dtest
# average_losses = []
# for e in capacities:
#     model = Regression_gradient(capacity=e,step_size= ,nb_steps= )
#     model.train(dataset)
#     #TODO
#
# plt.plot(capacities, average_losses, 'bo')
# plt.ylabel('average quadratic loss')
# plt.xlabel('lambda')
# plt.title('ridge regression with gradient descent')
# plt.show()