import numpy as np

x = np.allay([0,1])
w = np.allay([0.5,0.5])
b = -0.7
print(w*x)

print(np.sum(w*x))

print(np.sum(x*w)) + b
