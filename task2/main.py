import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np

import pandas as pd
import  matplotlib
column = ['debt', 'income', 'overdue']
# 查看数据集
data = pd.read_csv('E:\desk\credit-overdue.csv')
debt = data['debt'].values
data.columns = column
income = data['income'].values
overdue = data['overdue'].values
print(data['income'].isnull().sum())
print(data['debt'].isnull().sum())
print(data['overdue'].isnull().sum())
data_np = np.array(data)

print(data_np[:,2].sum())
plt.scatter(x=debt, y=income, c=overdue)

print(data.head())

#数据集划分
x_train, x_test, y_train, y_test = train_test_split(data[column[1:2]], data[column[2]], test_size=0.25, random_state=420)
L = LR()
params1 = {'penalty': ['l1', 'l2'], 'C': [x for x in range(1,5)], 'class_weight': ['balance'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
lg = GridSearchCV(L, params1, cv=10, scoring="f1")
lg.fit(x_train,y_train)
y_predict = lg.predict(x_test)
print("f1_score：", f1_score(y_test, y_predict))


lr = LR(C=0.9)
lr.fit(x_train,y_train)
y_predict2 = lr.predict(x_test)
print('准确值:', lr.score(x_test,y_test))
x1_plot = np.arange(1, 3, step=0.1)
x2_plot = -(x1_plot * lr.coef_[0][0] + lr.intercept_)
plt.plot(x1_plot, x2_plot)
print(lr.coef_)
# w = lg.coef_[0][0]
# x = np.linspace(0,1,5)
plt.show()









