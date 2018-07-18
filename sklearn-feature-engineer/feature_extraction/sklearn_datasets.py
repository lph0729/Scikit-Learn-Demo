#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-10 下午9:37
@email: lph0729@163.com  

"""
from sklearn.datasets import load_iris, load_boston

iris = load_iris()
boston = load_boston()

descr = iris.DESCR
# print(descr)

iris_feature_name = iris.feature_names
iris_target_name = iris.target_names

iris_feature_data = iris.data
iris_target_data = iris.target
print("-------------iris_datasets-------------------------------")
print("iris_feature_name:", iris_feature_name, "\niris_target_name:", iris_target_name, "\niris_data:",
      iris_feature_data, "\niris_target_data:", iris_target_data)

print("-------------boston_datasets-------------------------------")
boston_feature_name = boston.feature_names
# boston_target_name = boston.target_names # 没有目标值

boston_feature_data = boston.data  # (506, 13)
boston_target_data = boston.target  # (506,)

print("boston_feature_name:", boston_feature_name, "\nboston_feature_data:", boston_feature_data,
      "\nboston_target_data:", boston_target_data)
