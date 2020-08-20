import numpy as np

class Naive_Bayes:
    def __init__(self, X, Y):
        self.data = np.array(X)
        self.label = np.array(Y)
        self.classes = 2

    def fit(self):
        pc = {'pc_0':0,'pc_1':0}
        one_sample, zero_sample = [], []
        for i in range(len(self.label)):
            if self.label[i] == 0:
                zero_sample.append(self.data[i])
                pc['pc_0'] += 1
            else:
                one_sample.append(self.data[i])
                pc['pc_1'] += 1
        pc['pc_0'] /= len(self.label)
        pc['pc_1'] /= len(self.label)

        zero_sample = np.array(zero_sample)
        one_sample = np.array(one_sample)

        one_mean = []
        one_std = []
        zero_mean = []
        zero_std = []

        for i in range(np.shape(one_sample)[1]):
            one_tmp = one_sample[:,i].reshape(-1)
            one_mean.append(np.mean(one_tmp))
            one_std.append(np.std(one_tmp))

            zero_tmp = zero_sample[:,i].reshape(-1)
            zero_mean.append(np.mean(zero_tmp))
            zero_std.append(np.std(zero_tmp))

        self.pc = pc
        self.one_mean = one_mean
        self.one_std = one_std
        self.zero_mean = zero_mean
        self.zero_std = zero_std

    def gauss(self,x,mean,std):
        return (1/(std*(np.sqrt(2*np.pi)))) * np.exp(-1*((x-mean)**2/(2*std**2)))

    def predict(self,x):
        p1_comp, p0_comp = [], []
        for i in range(len(x)):
            p1_comp.append(self.gauss(x[i],self.one_mean[i],self.one_std[i]))
            p0_comp.append(self.gauss(x[i], self.zero_mean[i], self.zero_std[i]))

        p1, p0 = 1, 1
        for i in range(len(x)):
            p1 *= p1_comp[i]
            p0 *= p0_comp[i]
        p1 *= self.pc['pc_1']
        p0 *= self.pc['pc_0']

        return 1 if p1 > p0 else 0

    def evaluate(self, data, label):
        pred = []
        for i in range(len(label)):
            pred.append(self.predict(data[i]))

        count = 0
        for i in range(len(label)):
            if pred[i] == label[i]:
                count += 1
        print("Naive Bayes test acc: %f"%(count/len(label)))
        return pred
