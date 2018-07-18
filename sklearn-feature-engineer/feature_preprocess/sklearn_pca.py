#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-11 下午5:56
@email: lph0729@163.com  

"""
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = [[2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]]

pca = PCA()
result = pca.fit_transform(data)
print("feature:", result)

# 对鸢尾花数据进行降维
iris = load_iris()
pca = PCA(n_components=3)

iris_data = iris.data
result = pca.fit_transform(iris_data)

print("iris_data:", iris.data, "\npca_data:", result)


