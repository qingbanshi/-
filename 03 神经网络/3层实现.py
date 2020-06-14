import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:  # 老式神经网络用激活函数
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    c = x - np.sum(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    res = x / sum_exp_x

    return res


class Network_3():
    count = 0

    def __init__(self, **kwargs):
        Network_3.count += 1
        self.net = {"W1": kwargs["w1"],
                     "W2": kwargs["w2"],
                     "W3": kwargs["w3"],
                     "B1": kwargs["b1"],
                     "B2": kwargs["b2"],
                     "B3": kwargs["b3"]}

    def set_net(self, **kwargs):
        self.net["W1"] = kwargs["w1"]
        self.net["W2"] = kwargs["w2"]
        self.net["W3"] = kwargs["w3"]

        self.net["B1"] = kwargs["b1"]
        self.net["B2"] = kwargs["b2"]
        self.net["B3"] = kwargs["b3"]

    def start(self, X: np.ndarray, type=True):
        W1, W2, W3 = self.net["W1"], self.net["W2"], self.net["W3"]
        B1, B2, B3 = self.net["B1"], self.net["B2"], self.net["B3"]

        A1 = np.dot(X, W1) + B1
        Z1 = sigmoid(A1)

        A2 = np.dot(Z1, W2) + B2
        Z2 = sigmoid(A2)

        A3 = np.dot(Z2, W3) + B3
        Y = A3
        if type:
            return Y
        else:
            return softmax(Y)


w1 = np.array([[0.1, 0.3, 0.5],
               [0.2, 0.4, 0.6]])
w2 = np.array([[0.1, 0.6],
               [0.2, 0.5],
               [0.3, 0.4]])
w3 = np.array([[0.1, 0.3],
               [0.2, 0.4]])
b1 = np.array([0.1, 0.2, 0.3])
b2 = np.array([0.1, 0.2])
b3 = np.array([0.1, 0.2])

A = Network_3(w1=w1, w2=w2, w3=w3, b1=b1, b2=b2, b3=b3)
x = np.array([1.0, 0.5])

res = A.start(x)
