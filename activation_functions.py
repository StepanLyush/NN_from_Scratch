"""
This module contains commonly used activation functions for neural networks
"""

# import packages:
import numpy as np


# simple step function
def step(h, out=None):
    return np.where(h > 0, 1, 0)


# relu function:
def relu(h, out):
    return np.maximum(0, h, out=out)


# hyperbolic tangent (tanh) function:
def tanh(h, out):
    return np.tanh(h, out=out)


# sigmoid function:
def sigmoid(h, out):
    # if out:
    np.negative(h, out=out)
    np.exp(out, out=out)
    out += 1
    np.reciprocal(out, out=out)
    # else:
    #     out = 1 / (1 + np.exp(-h))
    return out


# softmax function:
def softmax(h, out=None):
    return np.exp(h)/np.exp(h).sum()


#%% Demonstration of the functions:

if __name__ == '__main__':
    # input data:
    x = np.array([0.3, 0.9])
    # weights:
    w1 = np.array([[0.3, 0.6], [0.1, 2], [0.9, 3.7], [0.0, 1]])
    w2 = np.array([0.75, 0.15, 0.22, 0.33])
    # bias:
    bias1 = np.array([0.7, 0.2, 1.1, 2])
    bias2 = 0.3
    # First hiddel layer:
    h1 = w1 @ x
    # Results:
    print('-'*50)
    print('Activation: Step function')
    print(f'Layer 1 output:\n{step(h1 - bias1)}\n\nNetwork output:\n{w2@step(h1 - bias1)}\n')
    print('-'*50)
    print('Activation: relu function')
    print(f'Layer 1 output:\n{relu(h1-bias1).round(decimals=2)}\n\nNetwork output:\n{(w2@relu(h1-bias1)).round(decimals=2)}\n')
    print('-'*50)
    print('Activation: tanh function')
    print(f'Layer 1 output:\n{tanh(h1-bias1).round(decimals=2)}\n\nNetwork output:\n{(w2@tanh(h1-bias1)).round(decimals=2)}\n')
    print('-'*50)
    print('Activation: sigmoid function')
    print(f'Layer 1 output:\n{sigmoid(h1-bias1).round(decimals=2)}\n\nNetwork output:\n{(w2@sigmoid(h1-bias1)).round(decimals=2)}\n')
    print('-'*50)


