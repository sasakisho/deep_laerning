# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import time

# データの読み込み
# x_train[60000][784],t_train[60000][10],x_test[10000][784],t_test[10000][10]
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 2層ニューラルネットワーク
# ニューロンの数(入力層:784, 隠れ層:50, 出力層:10)
# W1[784][50], W2[50][10], b1[50], b2[10]
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# ハイパーパラメータ:人に手により設定されるパラメータ。自動化できない
iters_num = 10000               # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]   # [60000,784][0] = 60000
batch_size = 100                # 一度に取り出す個数
learning_rate = 0.1             # 学習率:一回の学習でパラメータを変更する量

train_loss_list = []    # 損失関数の値を格納
train_acc_list = []     # 訓練データにおける認識精度を格納
test_acc_list = []      # テストデータにおける認識精度を格納

# 1エポックあたりの繰り返し数。
# 1エポック:訓練データを全て使い切った時のバッチ処理の繰り返し回数
# エポック:繰り返しの単位
iter_per_epoch = max(train_size / batch_size, 1)    #600

start = time.time()
for i in range(iters_num):
    # batch_mask[100], 0~59999のランダムな値が格納
    batch_mask = np.random.choice(train_size, batch_size)
    # [100][784], batch_maskをインデックスにしてデータの抽出、格納。
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)   #逆伝播、次章で解説

    # パラメータの更新
    # W1,W2,b1,b2を勾配の学習率倍で更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポック毎に認識精度の算出、格納。
    if i % iter_per_epoch == 0:
        # 訓練データとテストデータで認識精度を比較することで、
        # 過学習を起こしていないか確認できる。
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # tl = time.time()-start
    # print(str(i)+"回目"+str(tl)+" sec")

elapsed_time = time.time() - start
print("elapsed_time:"+ str(elapsed_time) + "[sec]")

# グラフの描画
markers = {'train': 'o', 'test': 's'}

# accuracy
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')

# loss
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 5.0)
plt.legend(loc='upper right')

plt.show()
