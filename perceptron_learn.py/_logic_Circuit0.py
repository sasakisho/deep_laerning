import numpy as np

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    theta = 0.7

    tmp = np.sum(w*x)
    #tmp = np.dot(w,x)

    if tmp <= theta:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    theta = -0.7

    tmp = np.sum(w*x)

    if tmp <= theta:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    theta = 0.2

    tmp = np.sum(w*x)

    if tmp <= theta:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

# def XOR(x1,x2):
#     x = np.array([x1,x2])
#     w1=np.array([[-0.5,-0.5],[0.5,0.5]])
#     w2=np.array([0.5,0.5])
#
#     s=np.dot(x,w1)
#     tmp=np.dot(s,w2)
#         if tmp <= theta:
#             return 0
#         else:
#             return 1
