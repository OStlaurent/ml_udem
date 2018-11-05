import numpy as np

np.random.seed(2)

circles = np.loadtxt('circles.txt')

# separate into train and test
n_classes = 2
n_train = 880
target_ind = [circles.shape[1] - 1]
inds = np.arange(circles.shape[0])
np.random.shuffle(inds)
train_inds = inds[:n_train]
test_inds = inds[n_train:]
train_set = circles[train_inds, :]
test_set = circles[test_inds, :]
train_inputs = train_set[:,:-1]
train_labels = train_set[:,-1]
test_inputs = test_set[:,:-1]
test_labels = test_set[:,-1]


class Neural_network:

    def __init__(self, d, dh, m, step_size, nb_iter):
        self.d = d # number of features
        self.dh = dh # number of neurons in the hidden layer
        self.m = m # number of class = number of neurons in the last layer (output layer)
        self.step_size = step_size
        self.nb_iter = nb_iter
        self.b1 = np.zeros(dh).reshape((dh,1))
        self.w1 = initialize_weight(d,dh)
        self.b2 = np.zeros(m).reshape((m,1))
        self.w2 = initialize_weight(dh,m)
        self.loss = 0

    def fprop(self,x,y = None):
        self.ha = self.b1 + np.dot(self.w1,x)
        self.hs = np.multiply(self.ha, (self.ha > 0))
        self.oa = self.b2 + np.dot(self.w2,self.hs)
        self.os = softmax(self.oa)
        if y is not None:
            self.loss += np.log(np.sum(np.exp(self.oa))) - os[y]

    def fprop_matrix(self,x,y = None):
        # TO VERIFY if correct and do the loss part
        #https://www.coursera.org/lecture/neural-networks-deep-learning/vectorizing-across-multiple-examples-ZCcMM
        x = x.T
        self.ha = self.b1 + np.dot(self.w1,x)
        self.hs = np.multiply(self.ha, (self.ha > 0))
        self.oa = self.b2 + np.dot(self.w2,self.hs)
        self.os = softmax(self.oa)
        if y is not None:
            self.loss += np.log(np.sum(np.exp(self.oa))) - os[y]


    def bprop(self,x,y):
        #TODO modifier pour que sa marche pour matrice
        grad_oa = self.os - np.eye(self.m)[y]
        grad_b2 = self.os - np.eye(self.m)[y]
        grad_w2 = np.outer(self.os,self.hs) - np.concatenate((np.zeros((y-1,self.dh)),self.hs.reshape((1,self.dh)),np.zeros((self.m-y,self.dh))))
        grad_hs = np.dot(self.w2.T,self.os) - self.w2[y,:].reshape((self.dh,1))
        vector_indicator = np.array([1 if e > 0 else 0 for e in np.nditer(self.hs)]).reshape(self.hs.shape)
        grad_ha = np.multiply(grad_hs,vector_indicator)
        grad_b1 = grad_ha
        #TODO verifier que grad_w1 donne la bonne chose
        grad_w1 = np.outer(grad_ha,x)

        #TODO ajouter gradient descent

    def train(self,train_data):
        pass

    def predict(self, x):
        self.fprop(x)
        return np.argmax(self.os)



def softmax(a):
    """
    numerically stable softmax
    :param a: vector dx1 or matrix dxn containing n vectors of length d
    :return: vector dx1 or matrix dxn
    """
    # find the max for each column
    max = np.max(a, axis=0)
    a = a - max
    a = np.exp(a)
    sum = np.sum(a, axis=0)
    a = a / sum
    return a

def initialize_weight(n_input,n2):
    """
    initialize weights of a layer with an uniform distribution [-1/n_input,1/n_input]
    n_input = number of inputs for the layer
    (bias is initialized to 0, so it's not initialize with this function)
    :param n_input: number of columns = number of neurons in the layer before = number of neurons that give inputs
    :param n2: number of rows = number of neurons in the layer 'after' or current layer = number of neurons that receive inputs
    :return: weights matrix
    """
    return np.random.uniform(-1/n_input,1/n_input,(n2,n_input))

def gradient_check():
    pass

a = np.array([1,4,3,2])
a2 = np.array([1,4,3,2,5,-9,0,2]).reshape((2,4))
a3 = np.array([1,4,3,2,10,40,30,20]).reshape((2,4))
grad_hs = np.dot(a2.T,a3) - a2[1,:].reshape((4,1))
print(grad_hs)
hs = np.array([-1,3,-1,3, 2,-2,2,-2]).reshape((4,2))
hs = np.array([-1,3,-1,3])
vector_indicator = np.array([1 if e > 0 else 0 for e in np.nditer(hs)]).reshape(hs.shape)
print(vector_indicator)
#print(np.multiply(grad_hs,vector_indicator))
print(np.multiply(grad_hs[:,1],vector_indicator))
