import numpy as np

class GDA:
    def __init__(self, X, Y):
        self.data = np.array(X)
        self.label = np.array(Y)
        self.phi = np.sum(self.label)/len(self.label)

    def fit(self):
        sample_1 = []
        sample_0 = []
        for i in range(len(self.label)):
            if self.label[i] == 1:
                sample_1.append(self.data[i])
            else:
                sample_0.append(self.data[i])
        self.mu_1 = np.mean(sample_1,axis=0)
        self.mu_0 = np.mean(sample_0,axis=0)

        N_1 = len(sample_1)
        N_0 = len(sample_0)

        self.sigma = np.zeros((np.shape(self.data[0])[0],np.shape(self.data[0])[0]))
        for i in range(len(self.label)):
            tmp = self.data[i].reshape(-1, 1)
            if self.label[i] == 1:
                mu = self.mu_1.reshape(-1,1)

            else:
                mu = self.mu_0.reshape(-1,1)

            self.sigma += np.matmul((tmp - mu), (tmp - mu).T)

        self.sigma /= (N_0 + N_1)

    def gauss(self,x,mu,sigma):
        p = len(x)
        a = 1/((2*np.pi)**(p/2) * np.linalg.det(sigma)**0.5)
        tmp_1 = np.matmul((x-mu).T, np.linalg.inv(sigma)**(-1))
        return a * np.exp(-0.5 * np.matmul(tmp_1, (x-mu)))

    def predict(self,x):
        p_1 = self.gauss(x, self.mu_1, self.sigma)
        p_1 = self.phi * p_1

        p_0 = self.gauss(x, self.mu_0, self.sigma)
        p_0 = self.phi * p_0

        return 1 if p_1 > p_0 else 0

    def evaluate(self, data, label):
        prediction = []
        count = 0
        for i in range(len(label)):
            pred = self.predict(data[i])
            prediction.append(pred)

            if pred == label[i]:
                count += 1
        print("GDA test acc: %f"%(count/len(label)))
        return prediction
