#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-12 下午4:54
@email: lph0729@163.com  

"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

x = iris.data
y = iris.target

mms = MinMaxScaler()
x = mms.fit_transform(x)

n_neighbors = [3, 5, 7, 9]

for n in n_neighbors:
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(x_train, y_train)

    scores = knn.score(x_test, y_test)
    y_pre = knn.predict(x_test)
    print("scores:\n", scores, "\ny_test:\n", y_test, "\ny_pred:\n", y_pre)
