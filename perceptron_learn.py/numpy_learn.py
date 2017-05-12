import numpy as np

a = np.array([1,2,3])
b = np.array([[4,5,6],[7,8,9]])

print(a)
print(b)
print("b[0] = "+str(b[0]))
print("b[:,0] = "+str(b[:,0]))

print("---.shape---")
print("a.shape = " + str(a.shape))
print("b.shape = " + str(b.shape))
print("b[0].shape = " + str(b[0].shape))
print("b[:,0].shape = " + str(b[:,0].shape))

print("---.reshape()---")
print("b.reshape(6) = " + str(b.reshape(6)))
print("b.reshape(3,2) = \r\n" + str(b.reshape(3,2)))

print("---np.randoum.randint(0,100,10)---")
c = np.random.randint(0,100,10)
print(c)

print("---四則演算---")
a = np.arange(4)
b = np.arange(9,5,-1)
print("a = " + str(a))
print("b = " + str(b))

print("a+b = "+ str(a+b))
print("a-b = "+ str(a-b))
print("a*b = "+ str(a*b))
print("a/b = "+ str(a/b))

print("---統計関数---")
print("np.amax(a) = " + str(np.amax(a)))
print("np.amin(a) = " + str(np.amin(a)))
print("np.mean(a) = " + str(np.mean(a)))
print("np.sum(a) = " + str(np.sum(a)))
print("np.std(a) = " + str(np.std(a)))
print("np.var(a) = " + str(np.var(a)))

print("---内積の計算---")
print("np.dot(a,b) = " + str(np.dot(a,b)))
