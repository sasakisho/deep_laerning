import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import matplotlib.pylab as plt
import gradient_simplenet as sn

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]))/batch_size

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 +0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1 - fxh2)/(2*h)
        x[idx]=tmp_val
    return grad

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(mean_squared_error(np.array(y),np.array(t)))
print(cross_entropy_error(np.array(y),np.array(t)))

(x_train,t_train) , (x_test,t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)

# x=np.arange(0.0,20.0,0.1)
# y=function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x,y)
# plt.show()

print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([5.0,7.0])))
print(numerical_gradient(function_2,np.array([12.0,0.0])))

net = sn.simpleNet()
print(net.W)

x=np.array([0.6,0.9])
p=net.predict(x)
print(p)

np.argmax(p)
t=np.array([0,0,1])
print(net.loss(x,t))

f=lambda w:net.loss(x,t)
dW = numerical_gradient(f,net.W)

print(dW)
