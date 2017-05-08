import nn_func as nf

x,t = nf.get_data()
network = nf.init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = nf.predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accracy_cnt+= 1
print("Accuracy:"+str(float(accuracy_cnt)/len(x)))
