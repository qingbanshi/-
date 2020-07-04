import numpy as np


class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr

    def Update(self, grads, params):
        for key in params.keys():
            params[key] -= grads[key]*self.lr


class Momentum:
    def __init__(self, lr = 0.01, alpha=.9):
        self.lr = lr
        self.v = None
        self.alpha = alpha

    def Update(self, grads, params):
        if self.v is None:
            self.v = {}
            for key,value in params.items():
                self.v[key] = np.zeros_like(value)

        for key in params.keys():
            self.v[key] = self.alpha*self.v[key]-self.lr*grads[key]
            params[key] += self.v[key]


class AdaGrad: # 可变学习率的方法
    def __init__(self,lr=.01):
        self.lr = lr
        self.h = None

    def Update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key,value in params.items():
                self.h[key] = np.zeros_like(value)
        for key in params.keys():
            self.h += grads[key]**2
            params[key] += self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)

