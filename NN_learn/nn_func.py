import numpy as np
import pickle

import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

# 重み付け、バイアスデータが入ったファイルの読み込み
# 第0層(入力層)784、第1層(隠れ層)50、第2層(隠れ層)100、第3層(出力層)10
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

# ニューラルネットワークの内積の計算→ソフトマックス関数値に変換
def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = softmax(a3)

    return y
