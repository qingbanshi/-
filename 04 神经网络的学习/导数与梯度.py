import numpy as np


def numerical_diff(f, x):
    delta = 1e-7
    return (f(x + delta) - f(x - delta)) / (2 * delta)


def numerical_geadient(f, x):
    """梯度运算,返回函数在此处的梯度值"""
    h = 1e-7
    grad = np.zeros_like(x)
    for idx in range(x.size):
        temp_val = x[idx]
        x[idx] = temp_val + h
        fxh1 = f(x)

        x[idx] = temp_val - h
        fxh2 = f(x)

        x[idx] = temp_val
        grad[idx] = (fxh1 - fxh2) / (2 * h)
    return grad


def gradient_descent(f, init_x, lr=.01, step_num=100):
    """lr即为学习率,用来控制梯度法行走次数"""
    x = init_x
    for i in range(step_num):
        grad = numerical_geadient(f, x)
        x -= grad*lr
    return x

