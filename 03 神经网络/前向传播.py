from dataset.mnist import load_mnist
import numpy as np
import pickle


def sigmoid(x):  # 老式神经网络用激活函数
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray):
    """概率函数,一般被省略,因为输出量太大,且不改变大小关系"""

    # 接下来是为了防止数据溢出采用的操作
    x -= np.sum(x)
    # 完(主要是利用上下同乘以以数,结果不变)(在下一步可以将加减便为乘除)

    exp_x = np.exp(x)  # 此处
    sum_exp_x = np.sum(exp_x)
    res = x / sum_exp_x
    return res


class Network3():
    def __init__(self):
        self.data = {}

    def set_net(self, **kwargs):
        self.data["W1"] = kwargs["w1"]
        self.data["W2"] = kwargs["w2"]
        self.data["W3"] = kwargs["w3"]

        self.data["B1"] = kwargs["b1"]
        self.data["B2"] = kwargs["b2"]
        self.data["B3"] = kwargs["b3"]

    def init_network(self):
        with open("a common net.pkl", "rb") as f:
            self.data = pickle.load(f)

    def start(self, X: np.ndarray, type=True):
        W1, W2, W3 = self.data["W1"], self.data["W2"], self.data["W3"]
        B1, B2, B3 = self.data["B1"], self.data["B2"], self.data["B3"]

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


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(True, True)
    return x_test, t_test


if __name__ == '__main__':
    x, t = get_data()
    accuracy_cnt = 0
    network = Network3()
    network.init_network()
    for i in range(len(x)):
        Y = network.start(x[i])
        p = np.argmax(Y)
        if p == t[i]:
            accuracy_cnt += 1

# TODO 记得删除lib中的common和database
