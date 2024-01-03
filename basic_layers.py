"""
This module contains basic layers of neural network models
"""

from typing import Protocol
from functions.activation_class import Relu, Sigmoid, Tanh
import numpy as np

# Protocol for layer parameters:
# For more info on protocols in python, see:
# https://www.pythontutorial.net/python-oop/python-protocol/
# https://towardsdev.com/define-interfaces-in-python-with-protocol-classes-4aea7e940e44


# Create a Layers protocol:
class Layers(Protocol):
    weights: np.ndarray
    bias: np.ndarray


# Create Backprop protocol:
class Backprop(Protocol):
    def forward(self) -> np.ndarray:
        pass

    def backward(self) -> np.ndarray:
        pass


# Create a protocol for Transfer:
class Transfer(Protocol):
    pass


# Create a fully connected inference layer class:
class FullyConnectedInferenceLayer:

    def __init__(self, w, b, activ_fn):
        self.weights = w
        self.bias = b
        self.activ_fn = activ_fn
        self.h = np.empty([1, w.shape[0]])

    def __call__(self, x):
        self.h = self.activ_fn((x @ self.weights - self.bias))
        return self.h

    # Alternative constructor (this is to be completed on Friday):
    # This function is equivalent to fully-connected clojure function from the 3rd blog post.
    @classmethod
    def from_dims(cls, activ_fn, in_dim, out_dim):
        w = np.empty([out_dim, in_dim])
        b = np.empty(out_dim)
        return cls(w, b, activ_fn)


# A separate function for alternative construction of a fully connected layer:
def fully_connected(activ_fn, in_dim, out_dim):
    w = np.empty([out_dim, in_dim])
    b = np.empty(out_dim)
    return FullyConnectedInferenceLayer(w, b, activ_fn)


# Create a fully connected learning layer class:
class FullyConnectedLearningLayer:

    def __init__(self, w, b, a_1, activ_fn):
        self._weights = w
        self._bias = b
        self.activ_fn = activ_fn
        self.a_1 = a_1
        self.z = np.empty([self._weights.shape[0], a_1.shape[1]])
        self.a = np.empty([self._weights.shape[0], a_1.shape[1]])

    def __call__(self, x):
        np.matmul(self._weights, x, out=self.z)
        np.subtract(self.z, self._bias, out=self.z)
        self.a = self.activ_fn.calculate(self.z, out=self.a)
        return self.a

    def forward(self):
        np.matmul(self._weights, self.a_1, out=self.z)
        np.subtract(self.z, self._bias, out=self.z)
        self.a = self.activ_fn.calculate(self.z, out=self.a)

    def backward(self):
        # Step 1:
        np.copyto(self.activ_fn.prime(self.z), self.z)
        np.multiply(self.z, self.a, out=self.z)
        # Step 2:
        v = np.matmul(self.a_1, self.z)
        # Step 3:
        np.matmul(self._weights.T, self.z, out=self.a_1)
        # Step 4:
        np.matmul(((self.z, np.ones([[self.z.shape[1], 1]]))/self.z.shape[1]), out=self.b)
        # Step 5:
        np.sum(self._weightss, v, out=self.w)

    def output(self):
        return self.a

    # Alternative constructor (this is to be completed on Friday):
    @classmethod
    def from_dims(cls, activ_fn, out_dim, previous):
        a_1 = previous.output()
        in_dim = int(a_1.shape[0])
        w = np.empty([out_dim, in_dim])
        b = np.empty([out_dim, 1])
        return cls(w, b, a_1, activ_fn)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        if new_weights.shape == self._weights.shape:
            self._weights = new_weights

    def initialize_weights(self, w):
        if w.shape == self._weights.shape:
            self._weights = w


# A separate function for alternative construction of a fully connected layer:
def fully_connected_learn(activ_fn, in_dim, out_dim, batch=16):
    w = np.empty([out_dim, in_dim])
    b = np.empty([out_dim])
    a_1 = np.empty([in_dim, batch])
    return FullyConnectedLearningLayer(w, b, a_1, activ_fn)


class InputLayer:

    def __init__(self, x):
        self.a = x

    def output(self):
        return self.a


#%% Code test and demonstration:

if __name__ == '__main__':

    # input:
    x0 = np.array([[0.3, 0.3, 0.3], [0.9, 0.9, 0.9]])
    # weights:
    w1 = np.array([[0.3, 0.6], [0.1, 2], [0.9, 3.7], [0.0, 1]])
    # w1 = np.array([[0.3, 0.1, 0.9, 0.0], [0.6, 2.0, 3.7, 1.0]])
    w2 = np.array([[0.75, 0.15, 0.22, 0.33]])
    # bias:
    bias1 = np.array([[0.7, 0.2, 1.1, 2]]).T
    bias2 = 0.3

    layer0 = InputLayer(x0)
    layer1 = FullyConnectedLearningLayer.from_dims(Tanh, 4, layer0)
    layer1.weights = w1
    layer1.bias = bias1
    layer2 = FullyConnectedLearningLayer.from_dims(Sigmoid, 1, layer1)
    layer2.weights = w2
    layer2.bias = bias2
    y = layer2(layer1(x0))
    print(layer1(x0))
    print(y.round(decimals=2))
    print(type(y))
    layer2.backward()

#%%


def mm(A, B, C, alpha, beta):
    np.sum(alpha*(np.matmul(A, B)), beta*C, out=C)

#%%
