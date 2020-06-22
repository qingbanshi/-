import numpy as np
from dataset.mnist import load_mnist
from common.functions import *
from common.gradient import numerical_gradient
import matplotlib.pylab as plt

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=.01):
        self.params = {}
        self.params["W1"] = weight_init_std*\
                            np.random.randn(input_size,hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)

    def predict(self,x):
        w1, w2 = self.params["W1"], self.params["W2"]
        B1, B2 = self.params["B1"], self.params["B2"]

        a1 = np.dot(x, w1)+B1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2)+B2

        y = softmax(a2)
        return y

    def loss(self, x, tag):
        y = self.predict(x)

        return cross_entropy_error(y, tag)

    def accuracy(self, x, t):
        Y =self.predict(x)
        y = np.argmax(Y, axis=1)
        t = np.argmax(t, axis=1)
        acc = np.sum(y==t)/float(x.shape[0])

        return acc

    def get_numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["B1"] = numerical_gradient(loss_W, self.params["B1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["B2"] = numerical_gradient(loss_W, self.params["B2"])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['B1'], self.params['B2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['B2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['B1'] = np.sum(dz1, axis=0)

        return grads


(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = .1
# 记录参数
train_losslist =[]

train_acclist =[]
test_acclist =[]
iter_per_epoch = max(train_size / batch_size, 1)

Network = TwoLayerNet(784, 50, 10)
for i in range(iters_num):
    # 取得一个mini_batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 获取梯度
    grad = Network.gradient(x_batch, t_batch)
    for key in ['W1','W2','B1','B2']:
        Network.params[key] -= learning_rate*grad[key]
    loss = Network.loss(x_batch, t_batch)
    train_losslist.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = Network.accuracy(x_train, t_train)
        test_acc = Network.accuracy(x_test, t_test)
        train_acclist.append(train_acc)
        test_acclist.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


x=np.arange(0,10000,1)
y = np.array(train_losslist)
plt.plot(x,y)
plt.show()
