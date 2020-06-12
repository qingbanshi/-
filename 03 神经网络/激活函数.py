import numpy as np
import matplotlib.pylab as plt


def step_function(x):  # 基于numpy数组的实现,
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):  # 老式神经网络用激活函数
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x:np.ndarray):
    """概率函数,一般被省略,因为输出量太大,且不改变大小关系"""

    # 接下来是为了防止数据溢出采用的操作
    x -= np.sum(x)
    # 完(主要是利用上下同乘以以数,结果不变)(在下一步可以将加减便为乘除)

    exp_x = np.exp(x) # 此处
    sum_exp_x = np.sum(exp_x)
    res = x / sum_exp_x
    return res


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1,1.1)
plt.show()


