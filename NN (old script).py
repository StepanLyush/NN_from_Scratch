# this is the inital script with all the code together. Other 3 files in the repo are splitted, more elaborated and completed code of the same logic.

import numpy as np


def threshold(array,
              thd):  # let's implement thing from graph from this article: https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-2-Bias-and-Activation-Function
    # array[array >= thd] = 1
    # array[array < thd] = 0
    return np.where(array >= thd, 1, 0)


x = np.array([0.3, 0.9])  # input vector
w1 = np.array([[0.3, 0.6],
               [0.1, 2],
               [0.9, 3.7],
               [0.0, 1]])  # weight matrix of the first layer
w2 = np.matrix([0.75, 0.15, 0.22, 0.33])

# we need to multiply weights matrix to the input vector (transposed) to get the output of the first layer:
l1 = np.matmul(w1, x.transpose())  # l1 stands for the output of the 1st layer
l2 = np.matmul(w2, l1)

# print(threshold(l2,1.836))

# let's implement different trasholds for different neurons of the first layer
# print('Layer 1: ', l1)
th1 = np.array([0.7, 0.2, 1.1, 2])
# print('Treshold 1 (biases): ', th1)
# print(np.greater_equal(l1, th1).astype('int8'))

# basiaclly, these are the biases - just tresholds:

bias1 = th1

# now let's implement RELU function instead:

# def relu(layer, treshold):
#     for i in range(len(layer)):
#         if layer[i] <= treshold[i]:
#             layer[i] = 0
#     return layer
#
# print(relu(l1, th1))

# basically we should first extract the tresholds (like l1-bias) -> then we can compare to 0 in function RELU

for i in range(len(l1)):
    l1[i] -= th1[i]


def relu(layer):
    for i in range(len(layer)):
        if layer[i] <= 0:
            layer[i] = 0
    return layer


# the more calculations we have the bigger figures we can reach. To reduce the calculations we can implement hyperbolic tangent (tanh)
# in this case we'll have figures from -1 to 1: (e2x-1/e2x+1)
def tanh(layer):
    return np.tanh(layer)


# print(tanh(l1, th1))

def sigmoid(layer):
    return 1 / (1 + np.exp(-layer))


# so let's make a class for layers:
class FullyConnectedInterface:
    def __init__(self, weights, bias=0, actfunc=relu):
        self.weights = weights
        self.bias = bias
        self.actfunc = actfunc

    def __str__(self):
        return f'weigh matrix -> {self.weights}; bias matrix -> {self.bias}; activation function -> {self.actfunc}'

    def __call__(self, prev_layer_output):
        self.layer_output = (self.weights @ prev_layer_output) - self.bias
        return self.actfunc(self.layer_output)


x = np.array([0.3, 0.9])
w1 = np.array([[0.3, 0.6], [0.1, 2], [0.9, 3.7], [0.0, 1]])
w2 = np.matrix([0.75, 0.15, 0.22, 0.33])

l1 = FullyConnectedInterface(w1, bias=bias1, actfunc=tanh)
l2 = FullyConnectedInterface(w2, bias=0.3, actfunc=sigmoid)
print(l2(l1(x)))

# articles 5 and 6 are not necessary for us, we skipped them

# we are interested in activation functions which derivatives are easily to compute


# let's make a class based on the previous one, but that stores the output values so we can make the backpropagation then

class FullyConnectedTrainingInterface:
    def __init__(self, weights, bias=0, actfunc=relu):
        self.weights = weights
        self.bias = bias
        self.actfunc = actfunc
        self.z = np.array([])       #z is the layer output before the activation
        self.a = np.array([])

    def __str__(self):
        return f'weigh matrix -> {self.weights}; bias matrix -> {self.bias}; activation function -> {self.actfunc}'

    def __call__(self, a_prev):
        self.z = (self.weights @ prev_layer_output) - self.bias     # z is the layer output without activation
        self.a = self.actfunc(self.z)                               # a is the layer output with activation
        return self.a

    @staticmethod
    def fully_connected(cls, in_dimension, out_dimension, actfunc, batch=1):     # out_dimension is basically the number of nodes in the leayer
        pass
        w = np.empty([out_dimension, in_demension])
        b = np.empty([out_dimension])
        return cls(w, b, actfunc)

l1 = FullyConnectedTrainingInterface(w1, bias=bias1, actfunc=tanh)
l2 = FullyConnectedTrainingInterface(w2, bias=0.3, actfunc=sigmoid)
print(l2(l1(x)))
print('First layer\'s output: ', l1.z, '; with activation: ', l1.a)
print('size: ', np.size(l1.weights, axis = 0), l1.weights)


# A good approach here now is to rewrite all the functions like:
# x = 1 / (1 + np.exp(-x))
# using approach below:
#     np.negative(x, out = x)
#     np.add(x, 1, out = x)
#     np.divide(1, x, out = x)
# What is it for? To answer the question we should check id(x) that will not change in the second approach.
# We can also run the next loops to compare the calculation time:
# import time
# t = time.time()
# for i in range(1000):
#     x = np.random.rand(1000*1000)
#     x = 1 / (1 + np.exp(-x))
# t = time.time() - t
# print(t)
# t = time.time()
# for i in range(1000):
#     x = np.random.rand(1000*1000)
#     np.negative(x, out = x)
#     np.add(x, 1, out = x)
#     np.divide(1, x, out = x)
# t = time.time() - t
# print(t)
# TODO: Also I should consider += and check if it is faster?

#%%
# 19.05.2023
# let's implement the same class but adding batches, in this case we need to have to define the input as columns not as rows by default in numpy
# in this case while we are creating the array with batches we must specify it with flag order = 'F' (F is for Fortran)
# to easlity compilate the derivative we can put it as another function or in activation function as an another return method
