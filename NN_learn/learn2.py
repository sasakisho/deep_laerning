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

# 重みとバイアスの生成
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])

    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])

    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

# ニューラルネットワークの内積
def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    A1 = np.dot(X,W1)+b1
    S1 = sigmoid(A1)

    A2 = np.dot(Z1,W2) + b2
    S2 = sigmoid(A2)

    A3 = np.dot(Z2,W3) + b3
    Y = identity_function(A3)

    return Y

# main
X = np.array([1.0,0.5])
network = init_network()
y = forward(network,X)
print(y)
