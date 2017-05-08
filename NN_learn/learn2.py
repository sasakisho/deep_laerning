import numpy as np

# a = np.array([[1,2],[3,4]])
# b = np.array([[4,3],[2,1]])
#
# print(np.dot(a,b))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
A1 = np.dot(X,W1)+B1
Z1 = sigmoid(A1)
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2,W3)
Y = identity_function(A3)

a =np.array([0.9,4.8,0.2])
y = softmax(a)
print(y)
print(np.sum(y))

#
# print(Z1)
# print(W2)
# print(B2)
# print(Y)
