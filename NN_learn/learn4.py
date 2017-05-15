import nn_func as nf
import numpy as np
import time

x,t = nf.get_data()
network = nf.init_network()

batch_size = 100
accuracy_cnt = 0

# start = time.time()

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = nf.predict(network,x_batch)

    p = np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

# elapsed_time = time.time() - start
# print("elapsed_time:"+ str(elapsed_time) + "[sec]")

print("Accuracy(正解率):"+str(float(accuracy_cnt)/len(x)))
