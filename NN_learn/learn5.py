import numpy as np

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 恒等関数
def identity_function(x):
    return x

#　ソフトマックス関数
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# main
X = np.array([1,2])
W = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
b = np.array([0.7,0.8,0.9])

y = identity_function(np.dot(X,W)+b)

hy = softmax(y)
#hy = sigmoid(y)

print("y=" + str(y))
print("hy=" + str(hy))
