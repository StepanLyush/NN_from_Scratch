"""
This module contains commonly used activation functions for neural networks
"""

# import packages:
import numpy as np


# simple step function
def step(h, out=None):
    return np.where(h > 0, 1, 0)


# relu function as a class:
class Relu:

    def calculate(x, out):
        return np.maximum(0, x, out=out)

    def prime(x):
        return np.where(x > 0, 1, 0)


class Sigmoid:

    def calculate(x, out):
        np.negative(x, out=out)
        np.exp(out, out=out)
        out += 1
        np.reciprocal(out, out=out)
        return out

    def prime(x):
        return np.sigmoid(x)*(1-np.sigmoid(x))


# hyperbolic tangent (tanh) function:
class Tanh:

    def calculate(x, out):
        np.tanh(x, out=out)

    def prime(x, out):
        np.tanh(x, out=out)
        np.square(out, out=out)
        np.subtract(1, out, out=out)


# softmax function:
def softmax(h, out=None):
    return np.exp(h)/np.exp(h).sum()


#%% Demonstration of the functions:

if __name__ == '__main__':
    # input data:
    x0 = np.array([0.3, 0.9])
    # weights:
    w1 = np.array([[0.3, 0.6], [0.1, 2], [0.9, 3.7], [0.0, 1]])
    w2 = np.array([0.75, 0.15, 0.22, 0.33])
    # bias:
    bias1 = np.array([0.7, 0.2, 1.1, 2])
    bias2 = 0.3
    # First hiddel layer:
    h1 = w1 @ x0
    temp = np.ones(h1.shape)
    # Tanh.prime(h1, temp)
    Tanh.prime(h1)

#%%

