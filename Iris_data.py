

import numpy as np

class Iris_Dataset:
    '''
    准备数据
    load_data:从文件中读取数据，组成特征向量
    sample_data:划分数据集，给定两种划分方式，两类（划分第一类，第二三类）、三类
    get_data: 获取按类划分并打乱好的数据
    split_data: 划分训练集 测试集
    '''
    def __init__(self, data_path = 'Iris_dataset/iris.data', classes = 2):
        self.data_path = data_path
        self.classes = classes

    def load_data(self):
        data = []
        name = []

        with open(self.data_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                a = line.replace('\n','').split(',')
                data.append(list(map(float,[x for x in a[:4]])))
                name.append(a[-1])

        Iris_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        label = []
        for i in range(len(data)):
            label.append(Iris_name.index(name[i]))
        self.all_data , self.all_label = data, label
        return data, label

    def sample_data(self):
        if self.classes == 2:
            data = self.all_data[:50]
            label = self.all_label[:50]

            # 在第二三类中采样50个作为第二类
            idx = np.arange(100)
            np.random.shuffle(idx)
            idx = idx[:50]
            idx += 50
            for i in idx:
                data.append(self.all_data[i])
                label.append(1)
        elif self.classes == 3:
            data = self.all_data
            label = self.all_label

        # 打乱
        sample_data = []
        sample_label = []
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        for i in idx:
            sample_data.append(data[i])
            sample_label.append(label[i])
        self.sample_data = sample_data
        self.sample_label = sample_label
        return sample_data, sample_label

    def get_data(self):
        self.load_data()
        data,label = self.sample_data()
        return np.array(data), np.array(label)

    def split_data(self, data, label):
        self.zero_count = 0
        self.one_count = 0

        test_data = []
        test_label = []
        train_data = []
        train_label = []
        index = 0
        while(self.zero_count < 10 or self.one_count < 10):
            if label[index] == 0:
                if self.zero_count < 10:
                    test_data.append(data[index])
                    test_label.append(label[index])
                    self.zero_count += 1
                else:
                    train_data.append(data[index])
                    train_label.append(label[index])

            elif label[index] == 1:
                if self.one_count < 10:
                    test_data.append(data[index])
                    test_label.append(label[index])
                    self.one_count += 1
                else:
                    train_data.append(data[index])
                    train_label.append(label[index])

            index += 1

        for i in range(index, len(data)):
            train_data.append(data[i])
            train_label.append(label[i])
        return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

#Iris = Iris_Dataset()
#data, label = Iris.get_data()
#train_data, train_label, test_data, test_label = Iris.split_data(data, label)


