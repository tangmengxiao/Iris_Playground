import numpy as np

# models
from method.Linear_Regression import Linear_Regression
from method.Perceptron import Perceptron
from method.LDA import LDA
from method.Logistic_Regression import Logistic_Regression
from method.Naive_Bayes import Naive_Bayes

# datasets
from Iris_data import Iris_Dataset

# 加载数据
Iris = Iris_Dataset()
data, label = Iris.get_data()
train_data, train_label, test_data, test_label = Iris.split_data(data, label)

# 建立模型
'''
# Linear Regression
model = Linear_Regression(train_data, train_label, L2=True, l=0.8)
model.fit()

y = model.predict(test_data[0])
predict = model.evaluate(test_data,test_label)

# Perceptron
model = Perceptron(train_data, train_label)
model.fit(epochs=100, )

predict = model.evaluate(test_data, test_label)


# LDA
model = LDA(train_data,train_label)
model.fit()
predict = model.evaluate(test_data,test_label)


# Logistic Regression
model = Logistic_Regression(train_data,train_label)
model.fit(epochs=20,lr=1e-3)
predict, predict_prob = model.evaluate(test_data,test_label)

'''

# Naive Bayes
model = Naive_Bayes(train_data, train_label)
model.fit()
predict = model.evaluate(test_data,test_label)
