#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-10 下午11:20
@email: lph0729@163.com  

"""
from sklearn.feature_extraction import DictVectorizer

data = [{'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}]

dv = DictVectorizer(sparse=False)

target = dv.fit_transform(data)

feature_name = dv.get_feature_names()
# target = dv.transform(data)
data_inverse = dv.inverse_transform(target)

print("feature_name:", feature_name, "\ntarget:", target, "\ndata_inverse:", data_inverse)


