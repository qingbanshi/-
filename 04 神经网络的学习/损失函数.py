import numpy as np

"""重要意义:精度可以连续的变化,以此来确定此函数梯度方向,进而确定参数变化方向
"""
def mean_squared_f(y,t):
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y,t):
    small = 1e-7  # 此处为了防止计算结果出现负无穷(ln(0))
    return -np.sum(t*np.log(y+small))

def cross_entropy_error_b(y:np.ndarray,t:np.ndarray):
    """当交叉熵函数输入为批数据时用此函数"""
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)/batch_size)

def cross_entropy_error_nom(y:np.ndarray,t:np.ndarray):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    tag_list =y[np.arange(batch_size), t] # 此处为了仅提取正确答案值
    res = -np.sum(t * np.log(tag_list + 1e-7) / batch_size)
    return res