import nn_func as nf
import numpy as np
import time

# x:入力,t:正解値
x,t = nf.get_data()
network = nf.init_network()

accuracy_cnt = 0

ac_box = np.zeros([10,10],dtype = np.int32)

# start = time.time()

for i in range(len(x)):#10000
    y = nf.predict(network,x[i])
    p = np.argmax(y)

    ac_box[p][t[i]] +=1

    if p == t[i]:
        accuracy_cnt+= 1

# elapsed_time = time.time() - start
# print("elapsed_time:"+ str(elapsed_time) + "[sec]")

print("Accuracy(正解率):"+str(float(accuracy_cnt)/len(x)))
print(ac_box)
