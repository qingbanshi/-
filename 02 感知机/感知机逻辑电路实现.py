import numpy as np


def main():

    def aand(x1, x2): # 与门
        w1, w2, theta = 0.3, 0.3, 0.5
        temp = x1*w1+x2*w2
        if temp>theta:
            return 1
        else:
            return 0
    print(aand(1,0))

    def new_and(x1, x2):  # 采用了偏置的方法
        a = np.array([x1, x2])
        w = np.array([.3,.3])
        b = -.5
        temp = np.sum(a*w)+b
        if temp > 0:
            return 1
        else:
            return 0

    def no_and(x1, x2): # 与非门(采用建议操作:将与门权重与偏置倒置)
        x = np.array([x1,x2])
        w = np.array([-.3,-.3])
        b = .5

if __name__ == '__main__':
    main()
