#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-13 下午2:32
@email: lph0729@163.com  

"""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd

data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

x = data[["pclass", "age", "sex"]]
y = data[["survived"]]

x["age"].fillna(data["age"].mean(), inplace=True)

x = pd.get_dummies(x, columns=["pclass", "sex"])
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 将数据集转化为字典，并且对有类别的数据特征变成one-hot编码
# x_dict = x_train.to_dict(orient="records")
# dc = DictVectorizer()
# x_train = dc.fit_transform(x_dict).toarray()
# feature_name = dc.get_feature_names()

dtc = DecisionTreeClassifier(criterion="entropy", max_depth=3)
dtc.fit(x_train, y_train)

y_pred = dtc.predict(x_test)
scores = dtc.score(x_test, y_test)

print("feature_data:\n", x, "\ntarget_data:\n", y, "\ny_pred:\n", y_pred, "\nscores:\n", scores)

export_graphviz(decision_tree=dtc, out_file="./tree.dot")

