import numpy as np

class play:
    def __init__(self):
        self.board=np.zeros([8,8])
        self.board[3,3]=1
        self.board[4,4]=1
        self.board[3,4]=-1
        self.board[4,3]=-1

    def check(self,x,y,color):
        if self.board[x][y]==0:
            return True
        else:
            return False

    def put(self,x,y,color):
        self.board[x,y]=color

    def func1(self,x,y):
        a=np.empty((0,2),int)
        for i in range(0,8):
            if (y-x+i>7 || y-x+i<0):
                a=np.append(a,np.array([[i,y-x+i]]),axis=0)
        return a

    # def func2(self,x,y):
    #     for i in range(0,8):
    #         if
    #     return
    #
    # def revarse(self,x,y,color):
