import numpy as np
import matplotlib.pylab as plt


def step_function(x):  # 基于numpy数组的实现,
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):  # 老式神经网络用激活函数
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1,1.1)
plt.show()


