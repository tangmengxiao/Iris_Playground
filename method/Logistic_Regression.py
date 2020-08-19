import numpy as np

class Logistic_Regression:
    def __init__(self, X, Y):
        # 数据预处理
        data = []
        for i in range(len(X)):
            # X[i]= append(1)
            tmp = X[i].tolist()
            tmp.append(1)
            data.append(tmp)

        self.data = np.array(data)
        self.data_dimension =  np.shape(self.data[0])[0]
        self.label = Y
        self.W = np.random.random((self.data_dimension,1))

    def sigmoid(self, x):
        x = x.reshape(-1,1)
        x = np.matmul(self.W.T,x)
        return 1/(1+np.exp(-x))

    def sign(self,x):
        return 1 if x >= 0.5 else 0

    def predict(self,x):
        return self.sign(np.matmul(self.W.T,x))

    def fit(self,epochs=100,lr=1e-2):
        for epoch in range(epochs):
            grad = np.zeros((self.data_dimension,1))
            for i in range(len(self.data)):
                grad += ((self.label[i] - self.sigmoid(self.data[i]))*self.data[i]).reshape(-1,1)
            self.grad = grad
            self.W += lr * grad

            # 测试
            count = 0
            self.prediction = []
            for i in range(len(self.data)):
                pred = self.predict(self.data[i])
                self.prediction.append(pred)
                if pred == self.label[i]:
                    count += 1

            #if epoch % 5 == 0:
            print(f'epoch: %d, acc: %f' % (epoch, (count / len(self.data))))

        return self.W

    def evaluate(self, X, Y):
        data = []
        for i in range(len(X)):
            # X[i]= append(1)
            tmp = X[i].tolist()
            tmp.append(1)
            data.append(tmp)
        data = np.array(data)

        predict = []
        predict_prab = []
        count = 0
        for i in range(len(data)):
            pred = self.predict(data[i])
            predict_prab.append(self.sigmoid(data[i]))
            predict.append(pred)
            if pred == Y[i]:
                count += 1
        print('Logistic Regression test acc: %f' % (count / len(Y)))
        return predict, predict_prab