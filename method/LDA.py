import numpy as np

class LDA:
    def __init__(self, X, Y):
        # 数据预处理
        self.data = np.array(X)

        # 标签预处理 将01标签转为 +-1标签
        for i in range(len(Y)):
            if Y[i] == 0:
                Y[i] = -1
        self.label = np.array(Y)

        x_c1 = []
        x_c2 = []
        for i in range(len(self.label)):
            if self.label[i] == 1:
                x_c1.append(self.data[i])
            else:
                x_c2.append(self.data[i])
        self.x_c1 = np.array(x_c1)
        self.x_c2 = np.array(x_c2)

    def fit(self):
        x_c1_bar = np.mean(self.x_c1, axis=0)
        x_c2_bar = np.mean(self.x_c2, axis=0)

        N1, N2 = len(self.x_c1), len(self.x_c2)

        S_c1 = np.zeros((len(x_c1_bar),len(x_c1_bar)))
        for i in range(N1):
            tmp_1 = self.x_c1[i].reshape(-1,1)
            tmp_2 = x_c1_bar.reshape(-1,1)
            res = np.matmul((tmp_1-tmp_2),(tmp_1-tmp_2).T)
            S_c1 += res
        S_c1 /= N1

        S_c2 = np.zeros((len(x_c2_bar), len(x_c2_bar)))
        for i in range(N2):
            tmp_1 = self.x_c2[i].reshape(-1,1)
            tmp_2 = x_c2_bar.reshape(-1,1)
            res = np.matmul((tmp_1-tmp_2),(tmp_1-tmp_2).T)
            S_c2 += res
        S_c2 /= N2

        S_w = S_c1 + S_c2

        self.W = np.matmul(np.linalg.inv(S_w),(x_c1_bar - x_c2_bar))
        return self.W

    def evaluate(self, X, Y):
        data = np.array(X)
        # 标签预处理 将01标签转为 +-1标签
        for i in range(len(Y)):
            if Y[i] == 0:
                Y[i] = -1
        label = np.array(Y)

        predict = []
        for i in range(len(label)):
            tmp_1 = data[i].reshape(-1,1)
            pred = np.matmul(self.W.T, tmp_1)
            if pred >= 0:
                predict.append(1)
            else:
                predict.append(-1)

        count = 0
        for i in range(len(label)):
            if predict[i] == label[i]:
                count += 1

        print('LDA test acc: %f' % (count / len(label)))
        return predict
