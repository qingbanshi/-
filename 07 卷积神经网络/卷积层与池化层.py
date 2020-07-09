import numpy as np
from common.util import im2col,col2im

class Convolution:
    """卷积层的算法实现

    # 初始化
    输入该层的 过滤器组--w,偏置组--b,步幅--stride,填充--pad

    #forward
    输入批数据
    输出运算结果

    #backward
    反向传播
    输入后一层传入数据(dout)
    向前一层输出(dx)

    """
    def __init__(self, w:np.ndarray, b:np.ndarray, stride=1, pad=0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

        self.col =None
        self.col_w = None

        self.dw = None
        self.db = None

    def forward(self,x:np.ndarray):
        FN,C,FH,FW = self.w.shape
        N,C,H,W = x.shape
        out_h = int((H+2*self.pad-FH)/self.stride+1) # 计算输出的长宽
        out_w = int((W+2*self.pad-FW)/self.stride+1)

        col = im2col(x,FH,FW,self.stride,self.pad) # 为了方便处理,转为2维矩阵
        col_W =self.w.reshape((FN,-1)).T # 展开滤波器

        out:np.ndarray = np.dot(col, col_W) + self.b # 正向传播计算
        out = out.reshape((N,out_h,out_w,-1)) # 根据层中输出形状整理数据
        out.transpose(0, 3, 1, 2)

        self.x = x   # 为反向传播存入数据
        self.col = col
        self.col_w = col_W
        return out

    def backward(self,dout:np.ndarray):
        FN,C,FH,FW = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1,FN)

        self.db = np.sum(dout,axis=0)
        self.dw = np.dot(self.col.T,dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout,self.col_w.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h:np.ndarray, pool_w:np.ndarray, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.argmax = None


    def forward(self, x:np.ndarray):
        N, C, H, W = x.shape
        out_h = int(1 + (H-self.pool_h)/self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col:np.ndarray = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)

        self.x=x
        self.argmax = np.argmax(col, axis=1)

        out = np.max(col,axis=1)
        out = out.reshape(N,out_h,out_w,C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout:np.ndarray):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_w*self.pool_h
        dmax = np.zeros(dout.size, pool_size)
        dmax[np.arange(self.argmax.size),self.argmax.flatten()] = dout.flatten()
        dmax:np.ndarray = dmax.reshape(dout.shape+(pool_size,))

        dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2],-1)
        dx = col2im(dcol,self.x.shape,self.pool_h,self.pool_w,self.stride,self.pad)

        return dx

