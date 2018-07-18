#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-17 下午10:42
@email: lph0729@163.com  

"""
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

iris = load_iris()

features_datas = iris.data

x_train, x_test, y_train, y_test = train_test_split(features_datas, iris.target)

k_means = KMeans(n_clusters=3)
k_means.fit(features_datas)

predict = k_means.predict(x_test)


colors = ["red", "green", "blue"]

plt.figure(figsize=(4, 4))
c = [colors[k] for k in predict]

plt.scatter(features_datas[:, 0], features_datas[:, 2], c=c)
plt.xlabel("feature_1")
plt.ylabel("feature_2")

plt.savefig("../k-means.png")

plt.show()


print("\ny_test:\n", y_test, "\npredict:\n", predict)
