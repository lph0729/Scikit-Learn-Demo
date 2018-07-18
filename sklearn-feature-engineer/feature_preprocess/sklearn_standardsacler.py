#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-11 下午4:05
@email: lph0729@163.com  

"""
from sklearn.preprocessing import StandardScaler

data = [[1., -1., 3.],
        [2., 4., 2.],
        [4., 6., -1.]]


ss = StandardScaler()

target = ss.fit_transform(data)

mean = ss.mean_
std = ss.var_
params = ss.get_params()

print("target:", target, "\nmean:", mean, "\nstd:", std, "\nparams:", params)
