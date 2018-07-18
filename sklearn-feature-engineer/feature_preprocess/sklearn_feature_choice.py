#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-11 下午5:43
@email: lph0729@163.com  

"""
from sklearn.feature_selection import VarianceThreshold

data = [[0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]]

vt = VarianceThreshold()

result = vt.fit_transform(data)
print("feature:", result)
