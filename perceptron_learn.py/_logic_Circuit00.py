# ANDゲート
def AND(x1,x2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.7

    # 信号の総和の計算
    tmp = x1*w1+w2*x2

    if tmp <= theta:
        return 0
    else:
        return 1





print("---AND回路---")
print("(0,0) -> " + str(AND(0,0)))
print("(0,1) -> " + str(AND(0,1)))
print("(1,0) -> " + str(AND(1,0)))
print("(1,1) -> " + str(AND(1,1)))
