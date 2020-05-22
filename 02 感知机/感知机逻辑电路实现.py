import numpy as np


def main():
    def aand(x1, x2):  # 与门
        w1, w2, theta = 0.3, 0.3, 0.5
        temp = x1 * w1 + x2 * w2
        if temp > theta:
            return 1
        else:
            return 0

    print(aand(1, 0))

    def new_and(x1, x2):  # 采用了偏置的方法
        a = np.array([x1, x2])
        w = np.array([.3, .3])
        b = -.5
        temp = np.sum(a * w) + b
        if temp > 0:
            return 1
        else:
            return 0

    def no_and(x1, x2):  # 与非门(采用建议操作:将与门权重与偏置倒置)
        x = np.array([x1, x2])
        w = np.array([-.3, -.3])
        b = .5
        temp = np.sum(x * w) + b
        if temp > 0:
            return 1
        elif temp <= 0:
            return 0

    def Or(x1, x2):
        x = np.array([x1, x2])
        w = np.array([.7, .7])
        b = -.6
        temp = np.sum(w * x) + b
        if temp > 0:
            return 1
        elif temp <= 0:
            return 0

    # 由于感知机本身的局限性所致,无法通过简单的数值设计来进行异或门设计,因此采用逻辑门组合的方式来实现
    def Noor(x1, x2):
        t1 = Or(x1, x2)
        t2 = no_and(x1, x2)
        return aand(t1,t1)
# 通过层间叠加,感知机可以实现非线性函数的表达效果,可以构建完整计算机


if __name__ == '__main__':
    main()
