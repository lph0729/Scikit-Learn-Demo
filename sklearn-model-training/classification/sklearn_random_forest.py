#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-16 上午9:49
@email: lph0729@163.com  

"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

x_data = data[["pclass", "sex", "age"]]
x_data["age"].fillna(data["age"].mean(), inplace=True)
x = pd.get_dummies(x_data, columns=["pclass", "sex"])  # 将pclass age 两个特征是转化为one-hot编码

y = data[["survived"]]

x_train, x_test, y_train, y_test = train_test_split(x, y)

rfc = RandomForestClassifier(n_estimators=5, criterion="entropy", bootstrap=True, max_depth=5)
rfc.fit(x_train, y_train)

score = rfc.score(x_test, y_test)
y_predict = rfc.predict(x_test)

report = classification_report(y_true=y_test, y_pred=y_predict)


print("x:\n", x[:10], "\ny:\n", y[:10], "\nscore:\n", score, "\ny_predict:\n", y_predict, "\nreport:\n", report)
