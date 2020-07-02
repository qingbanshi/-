import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class OpTwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        self.params = {}
        self.params["W1"] = weight_init_std*np.random.randn(input_size,hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(hidden_size)

        self.layers = OrderedDict() # 有序字典 为了正反向传播的顺序问题
        self.layers["ReLu"] = Relu()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["B1"])
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["B2"])
        self.LLayer = SoftmaxWithLoss()

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):

        Y = self.predict(x)

        return self.LLayer.forward(Y, t)

    def accuracy(self, x, t):
        Y = self.predict(x)
        y = np.argmax(Y, axis=1)
        if t.ndim != 1 : t= np.argmax(t, axis=1) # 为了进行小批量转换(ndim=维度)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """传统的接口    使用梯度计算 相当于求偏导"""
        grads = {}
        loss_W = lambda W: self.loss(x, t)
        grads["dW1"] = numerical_gradient(loss_W,self.params["W1"])
        grads["dB1"] = numerical_gradient(loss_W,self.params["B1"])
        grads["dW2"] = numerical_gradient(loss_W,self.params["W2"])
        grads["dB2"] = numerical_gradient(loss_W,self.params["B2"])

        return grads

    def gradient(self, x, t):
        """反向传播接口 直接使用传播计算 提高效率 相当于仅求值"""
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.LLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout  = layer.backward(dout)

        # datas
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["B1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads