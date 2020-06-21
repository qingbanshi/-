from common.functions import *
from common.gradient import numerical_gradient
import numpy as np


class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=.01):
        self.params = {}
        self.params["W1"] = weight_init_std*\
                            np.random.randn(input_size,hidden_size),
        self.params["B1"] = np.zeros(hidden_size),
        self.params["W2"] = weight_init_std * \
                            np.random.randn(hidden_size, output_size),
        self.params["B2"] = np.zeros(hidden_size)

    def predict(self,x):
        w1, w2 = self.params["W1"], self.params["W2"]
        B1, B2 = self.params["B1"], self.params["B2"]

        a1 = np.dot(x, w1)+B1
        a2 = np.dot(a1, w2)+B1

        y = softmax(a2)
        return y

    def loss(self, x, tag):
        y = self.predict(x)

        return cross_entropy_error(y,tag)

    def accuracy(self, x, t):
        Y =self.predict(x)
        y = np.argmax(Y, axis=1)
        t = np.argmax(t, axis=1)
        acc = np.sum(y==t)/float(x.shape[0])

        return acc

    def numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["B1"] = numerical_gradient(loss_W, self.params["B1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["B2"] = numerical_gradient(loss_W, self.params["B2"])

        return grads

