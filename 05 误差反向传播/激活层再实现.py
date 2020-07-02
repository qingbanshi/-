import numpy as np
class RuLu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x[:]
        out[self.mask] = 0

        return out

    def backward(self,dout):
        dout[self.mask]=0

        return dout


class Sigmoid:
    def __init__(self):
        self.data = None

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.data = out
        return  out

    def backward(self,out):
        dout = out*self.data*(1.0-self.data)

        return dout
