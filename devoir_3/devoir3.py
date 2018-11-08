import numpy as np

np.random.seed(3)

circles = np.loadtxt('circles.txt')

#from tp
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

    def __init__(self, d, dh, m, epoch = 100, step_size= 0.01, k = 1, lambda11 = 0, lambda12 = 0, lambda21 = 0, lambda22 = 0):
        self.d = d # number of features
        self.dh = dh # number of neurons in the hidden layer
        self.m = m # number of class = number of neurons in the last layer (output layer)
        self.step_size = step_size
        self.epoch = epoch
        self.b1 = np.zeros((dh,1))
        self.w1 = initialize_weight(d,dh)
        self.b2 = np.zeros((m,1))
        self.w2 = initialize_weight(dh,m)
        self.k = k  # size of minibatch
        self.lambda11 = lambda11
        self.lambda12 = lambda12
        self.lambda21 = lambda21
        self.lambda22 = lambda22
        self.loss = 0

    def weight_decay(self,w):
        if w == 'w1':
            return self.lambda11*np.sign(self.w1) + 2*self.lambda12*self.w1
        elif w == 'w2':
            return self.lambda21*np.sign(self.w2) + 2*self.lambda22*self.w2

    def fprop(self,x,y = None):
        #TODO fix loss or remove it
        self.ha = self.b1 + np.dot(self.w1,x)
        self.hs = self.ha.clip(min=0)
        self.oa = self.b2 + np.dot(self.w2,self.hs)
        self.os = softmax(self.oa)
        # if y is not None:
        #     #TODO add regulation
        #     #correct equation?
        #     self.loss += np.log(np.sum(np.exp(self.oa))) - self.os[y]


    def bprop(self,x,y):
        grad_b2 = self.os - np.eye(self.m)[y].reshape((self.m,1))
        # grad_w2 = np.outer(self.os,self.hs) - np.concatenate((np.zeros((y-1,self.dh)),self.hs.reshape((1,self.dh)),np.zeros((self.m-y,self.dh))))
        grad_w2 = np.outer(self.os, self.hs)
        for i in range(self.hs.shape[0]):
            grad_w2[y][i] - self.hs[i]
        grad_hs = np.dot(self.w2.T,self.os) - self.w2[y,:].reshape((self.dh,1))
        indicator = np.sign(self.hs).clip(min=0)
        grad_ha = np.multiply(grad_hs,indicator)
        grad_b1 = grad_ha
        grad_w1 = np.outer(grad_ha,x)
        return np.array([grad_b1, grad_w1, grad_b2, grad_w2])

    def gradient_descent(self,grad_b1, grad_w1, grad_b2, grad_w2):
        self.w1 = self.w1 - self.step_size*(grad_w1 + self.weight_decay('w1'))
        self.b1 = self.b1 - self.step_size*grad_b1
        self.w2 = self.w2 - self.step_size * (grad_w2 + self.weight_decay('w2'))
        self.b2 = self.b2 - self.step_size * grad_b2

    def gradient_check(self,x,y):
        epsilon = 0.0001
        #y = y.astype(int)
        x = x.reshape((self.d, 1))
        self.fprop(x,y)
        f1 = np.array([self.b1, self.w1, self.b2, self.w2])
        loss1 = - np.log(self.os[y])
        self.b1 = self.b1 - epsilon
        self.w1 = self.w1 - epsilon
        self.b2 = self.b2 - epsilon
        self.w2 = self.w2 - epsilon
        self.fprop(x, y)
        f2 = np.array([self.b1, self.w1, self.b2, self.w2])
        loss2 = - np.log(self.os[y])

        self.b1 = self.b1 + 2*epsilon
        self.w1 = self.w1 + 2*epsilon
        self.b2 = self.b2 + 2*epsilon
        self.w2 = self.w2 + 2*epsilon
        #estimate = (loss1 - loss2)/epsilon
        #print(estimate)
        estimate = f1 - f2 / epsilon
        computed = self.bprop(x,y)
        print(computed/estimate)

    # def features_dimension_check(self, train_inputs):
    #     #correct the network if the number of features doesnt match with the number of neurons in the input layer
    #     if train_inputs.shape[1] != self.d:
    #         self.__init__(train_inputs.shape[1], self.dh, self.m, self.nb_iter, self.step_size, self.k, self.lambda11,
    #                       self.lambda12, self.lambda21, self.lambda22)

    def train(self,train_inputs, train_labels):
        # self.features_dimension_check(train_inputs)
        train_labels = train_labels.astype(int)
        for l in range(self.epoch):
            #separate the dataset in minibatch with k examples
            for i in range(train_inputs.shape[0]//self.k):
                s = i * self.k #start index of the current minibatch
                e = (i + 1) * self.k #end index of the current minibatch
                self.train_minibatch( minibatch_inputs = train_inputs[s:e, :], minibatch_labels  = train_labels[s:e])

            #if the numbers of examples in the trainning dataset is not a multiple of k
            # take care of the last minibatch that cant have k examples
            if (train_inputs.shape[0] % self.k) != 0:
                s = (train_inputs.shape[0] // self.k) * self.k
                self.train_minibatch(minibatch_inputs = train_inputs[s:, :], minibatch_labels = train_labels[s:])

    def train_minibatch(self, minibatch_inputs, minibatch_labels):
        total_b1 = np.zeros((self.dh,1))
        total_w1 = np.zeros((self.dh,self.d))
        total_b2 = np.zeros((self.m,1))
        total_w2 = np.zeros((self.m,self.dh))
        grads = np.array([total_b1, total_w1, total_b2, total_w2])
        #grads = np.array([total_b1, total_w1, total_b2, total_w2]).reshape((4,1))

        # sum gradient from the k examples
        for j in range(minibatch_inputs.shape[0]):
            input = minibatch_inputs[j].reshape((self.d,1))
            self.fprop(input, minibatch_labels[j])
            grads += self.bprop(input, minibatch_labels[j])

        # use average gradients of the minibatch
        grads = grads / minibatch_inputs.shape[0]
        self.gradient_descent(grads[0], grads[1], grads[2], grads[3])

    def predict(self, x, y):
        x =  x.reshape((self.d, 1))
        self.fprop(x)
        # print(self.os, y)
        return float(np.argmax(self.os))


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


model = Neural_network(2,4,2, epoch=10, step_size=0.01, lambda11=0,lambda12=0,lambda21=0,lambda22=0)
model.train(train_inputs, train_labels)

# test_inputs[0:3] == [[-0.42091717, -0.68031517],[-0.97401192, -0.22649677],[ 0.7444175,  -0.66771445]]
# test_labels[0:3] = [1., 0., 0.]

# print(model.b1)
b1 = model.b1
# print('_________________')
# print(model.w1)
w1 = model.w1
# print('_________________')
# print(model.b2)
b2 = model.b2
# print('_________________')
# print(model.w2)
w2 = model.w2
# print('_________________')

x = test_inputs[0]
x =  x.reshape((2, 1))

y = int(test_labels[0])

model.gradient_check(x,y)


#fprop
ha = b1 + np.dot(w1, x)
#print(b1)
#print(np.dot(w1, x), 'w1*x')
#print(w1, 'w1')
#print(ha)
hs = ha.clip(min=0)
#print(hs, 'hs')
#print(w2)
#print(b2)
#print(np.dot(w2, hs), 'w2*hs')
oa = b2 + np.dot(w2, hs)
#print(oa)
os = softmax(oa)
#print(os)


step_size = 0.01
#bprop
grad_b2 = os - np.eye(2)[y].reshape((2, 1))
#print(grad_b2)
# grad_w2 = np.outer(self.os,self.hs) - np.concatenate((np.zeros((y-1,self.dh)),self.hs.reshape((1,self.dh)),np.zeros((self.m-y,self.dh))))
grad_w2 = np.outer(os, hs)
for i in range(hs.shape[0]):
    grad_w2[y][i] - hs[i]
grad_hs = np.dot(w2.T, os) - w2[y, :].reshape((4, 1))
# vector_indicator = np.array([1 if e > 0 else 0 for e in np.nditer(self.hs)]).reshape(self.hs.shape)
indicator = np.sign(hs).clip(min=0)
grad_ha = np.multiply(grad_hs, indicator)
grad_b1 = grad_ha
grad_w1 = np.outer(grad_ha, x)
#return np.array([grad_b1, grad_w1, grad_b2, grad_w2])

#gradient descent
# w1 = w1 - self.step_size * (grad_w1 + self.weight_decay('w1'))
# b1 = b1 - self.step_size * grad_b1
# w2 = w2 - self.step_size * (grad_w2 + self.weight_decay('w2'))
# b2 = b2 - self.step_size * grad_b2

x = test_inputs[1]
x =  x.reshape((2, 1))
ha = b1 + np.dot(w1, x)
#print(np.dot(w1, x), 'w1*x')
#print(w1, 'w1')
#print(ha)
hs = ha.clip(min=0)
#print(hs, 'hs')
#print(w2)
#print(np.dot(w2, hs), 'w2*hs')
#print(b2)
oa = b2 + np.dot(w2, hs)
#print(oa)
os = softmax(oa)
#print(os)






## Calcul du pourcentage d'exemple qui se fait correctement classifier
# correct = 0
# for i in range(test_inputs.shape[0]):
#     if model.predict(test_inputs[i],test_labels[i]) == test_labels[i]:
#         correct = correct + 1
# print('Taux de classification correcte:',correct / test_inputs.shape[0], '%')