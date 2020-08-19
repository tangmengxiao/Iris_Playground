import numpy as np

from Iris_data import Iris_Dataset

class Linear_Regression:
    '''
    初始化: 数据
    建模
    评定
    '''
    def __init__(self,X, Y, L2=False, l=0.2):
        data = []
        for i in range(len(X)):
            #X[i]= append(1)
            tmp = X[i].tolist()
            tmp.append(1)
            data.append(tmp)

        self.X = np.array(data)
        self.Y = Y
        self.L2 = L2
        self.l = l

    def fit(self):
        if self.L2 == False:
            tmp = np.linalg.inv(np.matmul(self.X.T,self.X))
            tmp = np.matmul(tmp, self.X.T)
            self.W = np.matmul(tmp, self.Y)
            return self.W

        else:
            tmp1 = np.matmul(self.X.T,self.X)
            m,n = np.shape(tmp1)
            I = np.ones((m,n))
            tmp2 = tmp1 + self.l * I
            tmp2 = np.linalg.inv(tmp2)
            tmp3 = np.matmul(tmp2, self.X.T)
            self.W = np.matmul(tmp3, self.Y)
            return self.W

    def predict(self,x):
        x = x.tolist()
        x.append(1)
        x = np.array(x)
        return np.matmul(self.W.T, x)

    def evaluate(self,X,Y):
        predict = []
        for i in X:
            predict.append(self.predict(i))

        count = 0
        for i in range(len(X)):
            if predict[i] >= 0.5 and Y[i] == 1:
                count += 1
            elif predict[i] < 0.5 and Y[i] == 0:
                count += 1

        print("Linear Regression test acc: %f"%(count/len(Y)))
        return predict


