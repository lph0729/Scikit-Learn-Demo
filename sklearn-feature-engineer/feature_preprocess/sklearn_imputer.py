#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-11 下午5:22
@email: lph0729@163.com  

"""
from sklearn.preprocessing import Imputer
import numpy as np

data = [[1, 2],
        [np.nan, 3],
        [7, 6]]

imputer = Imputer(missing_values="NaN", axis=1, strategy="mean")

result = imputer.fit_transform(data)

print("target:", result)

