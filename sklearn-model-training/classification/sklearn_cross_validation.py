#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-12 下午5:31
@email: lph0729@163.com  

"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# 1.数据的加载以及预处理
file_path = "/media/payneli/工作文件/python_2017/Couresewares--html版/人工智能------doc版/data/FBlocation/train.csv"
data = pd.read_csv(file_path).query("3< x<4& 3< y < 4")
data_place_id = data.groupby("place_id").agg(np.count_nonzero).query("row_id > 3").reset_index()

model_data = data[data["place_id"].isin(data_place_id["place_id"])]

# 2.数据集划分
x = model_data.drop(["place_id", "time", "row_id"], axis=1)
y = model_data["place_id"]
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 3.数据集特征的标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 4.使用GridSearchCV与KNN进行模型训练
knn = KNeighborsClassifier(n_neighbors=5)

param_grid = {"n_neighbors": [3, 5, 7]}
gscv = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3)
gscv.fit(x_train, y_train)

# 5.模型评估以及测试数据预测
best_scores = gscv.best_score_
best_estimator = gscv.best_estimator_
cv_results = gscv.cv_results_

y_pred = gscv.predict(x_test)

print("best_scores:", best_scores, "\nbest_estimator:", best_estimator, "\ncv_results:", cv_results,
      "\ny_pred:", y_pred)
