#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-12 上午10:43
@email: lph0729@163.com  

"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 1.数据的加载以及预处理
file_path = "/media/payneli/工作文件/python_2017/Couresewares--html版/人工智能------doc版/data/FBlocation/train.csv"
data = pd.read_csv(file_path).query("3<x<4 & 3<y<4")
place_id_count = data.groupby("place_id").agg(np.count_nonzero)
renew_data = place_id_count.query("row_id > 3").reset_index()
# renew_data = place_id_count[place_id_count["row_id"] > 3].reset_index()
model_data = data[data["place_id"].isin(renew_data["place_id"])]

# 2.数据集的划分
x = model_data.drop(["place_id", "row_id", "time"], axis=1)
y = model_data["place_id"]

x_train, x_test, y_train, y_test = train_test_split(x, y)

# 3.数据集进行标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

n_neighbors = [3, 5, 7, 9, 11, 13, 15]

for n in n_neighbors:
    # 4.使用KNN进行模型训练
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train, y_train)

    # 5.模型的准确度以及预测目标值
    scores = knn.score(x_test, y_test)
    y_pre = knn.predict(x_test)

    print("data:\n", data[0:10], "\nplace_id_count:\n", place_id_count[:5], "\nrenew_data:\n", renew_data[:5],
          "\nmodel_data:\n", model_data, "scores:\n", scores, "\ny_pre:\n", y_pre)
