from common import layers
import numpy as np
from collections import OrderedDict

def conv_output_size(args):
    pass


class SimpleConvNet:
    def __init__(self, input_dim=(1,28,28), hidden_size=100,
                 output_size=10, weight_init_std=.01,
                 conv_param = {"filter_num":30, "filter_size":5, "pad":0
                 "stride":1}):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["filter_pad"]
        filter_stride= conv_param["filter_stride"]
        input_size = input_dim[1]
        conv_output_size = (input_size-filter_size+2*filter_pad)/\
            filter_stride + 1
        pool_output_size = int(filter_num*(conv_output_size**2)/4)

        self.parmas = {}
        self.parmas["w1"] = weight_init_std*np.random.randn(filter_num, input_dim[0],
                                                            filter_size, filter_size)
        self.parmas["b1"] = np.zeros(filter_num)


        self.parmas["w2"] = weight_init_std * np.random.randn(pool_output_size,
                                                              hidden_size)
        self.parmas["b2"] = np.zeros(hidden_size)


        self.parmas["w3"] = weight_init_std * np.random.randn(hidden_size,
                                                              output_size)
        self.parmas["b3"] = np.zeros(output_size)



        self.layers = OrderedDict()
        self.layers["Conv1"] = layers.Convolution(self.parmas["w1"],
                                                  self.parmas["b1"],
                                                  filter_stride,
                                                  filter_pad)
        self.layers["Relu1"] = layers.Relu()
        self.layers["pool1"] = layers.Pooling(pool_h=2,pool_w=2,stride=2)
        self.layers["Affine1"] = layers.Affine(self.parmas["w2"],
                                               self.parmas["b2"])
        self.layers["Relu2"] = layers.Relu()
        self.layers["Affine2"] = layers.Affine(self.parmas["w3"],
                                               self.parmas["b3"])
        self.lastlayer = layers.SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastlayer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        rlayers = list(self.layers.values())
        rlayers.reverse()

        for layer in rlayers:
            dout = layer.backward(dout)

        grads = {}

        grads["w1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["w2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["w3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db

        return grads
