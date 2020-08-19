import numpy as np

class Perceptron:
    '''
    二分类感知机
    '''
    def __init__(self, X, Y):
        # 数据预处理
        data = []
        for i in range(len(X)):
            # X[i]= append(1)
            tmp = X[i].tolist()
            tmp.append(1)
            data.append(tmp)

        self.data = np.array(data)

        # 标签预处理 将01标签转为 +-1标签
        for i in range(len(Y)):
            if Y[i] == 0:
                Y[i] = -1
        self.label = np.array(Y)
        self.W = np.random.random(np.shape(data[0]))

    def sign(self,x):
        return 1 if x >= 0 else -1

    def predict(self,x):
        return self.sign(np.matmul(self.W.T,x))

    def fit(self, epochs=500, lr=0.001):

        for epoch in range(epochs):
            self.grad = 0

            for i in range(len(self.data)):
                pred = self.predict(self.data[i])

                # 分类错误的
                if pred != self.label[i]:
                    self.grad += self.label[i] * self.data[i]

            self.W = self.W + lr * self.grad

            # 测试
            count = 0
            for i in range(len(self.data)):
                pred = self.predict(self.data[i])
                if pred == self.label[i]:
                    count += 1

            if epoch % 5 == 0:
                print(f'epoch: %d, acc: %f'%(epoch,(count/len(self.data))))
        return self.W

    def evaluate(self, X, Y):
        data = []
        for i in range(len(X)):
            # X[i]= append(1)
            tmp = X[i].tolist()
            tmp.append(1)
            data.append(tmp)

        # 标签转化
        for i in range(len(Y)):
            if Y[i] == 0:
                Y[i] = -1
        predict = []
        count = 0
        for i in range(len(data)):
            pred = self.predict(data[i])
            predict.append(pred)
            if pred == Y[i]:
                count += 1
        print('Preceptron test acc: %f'%(count/len(Y)))
        return predict