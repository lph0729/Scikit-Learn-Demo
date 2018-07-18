#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-11 下午3:50
@email: lph0729@163.com  

"""
from sklearn.preprocessing import MinMaxScaler

data = [[90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]]

mms = MinMaxScaler()
result = mms.fit_transform(data)

min = mms.data_min_
max = mms.data_max_
avg = mms.data_range_

print("normalization:", result, "\n min:", min, "\nmax:", max, "\navg:", avg)
