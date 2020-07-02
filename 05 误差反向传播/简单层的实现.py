import numpy as np
from common.functions import *

# 加法层
class PluLayer:

    def forword(self, x, y):
        out = x+y
        return out

    def backword(self, out):
        dx = out
        dy = out

        return  dx,dy


# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forword(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backword(self, out):
        dx = out*self.y
        dy = out*self.x

        return  dx,dy


# 传播层(既中间的计算层)
class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dw = None
        self.db = None
        self.x = None
    def forward(self, x):
        self.x = x
        Y = np.dot(self.x, self.W)+self.b

        return  Y

    def backward(self, dout):

        self.dw = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)
        self.db = np.sum(dout,axis=0)

        return dx

class SoftMaxWithLoss:
    def __init__(self):
        self.x = None
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.x = x
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.x)/batch_size

        return dx

    