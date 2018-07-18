#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-10 下午10:39
@email: lph0729@163.com  

"""
from sklearn.feature_extraction.text import CountVectorizer

"""需要注意点：fit_transform = fit + transform"""

# data = ["life is short, i like python", "life is too long, i dislike python"]
data = ["All Python releases are Open Source. Historically, most, but not all, Python releases have also been "
        "GPL-compatible. The Licenses page details GPL-compatibility and Terms and Conditions.",
        "For most Unix systems, you must download and compile the source code. The same source code archive can also "
        "be used to build the Windows and Mac versions, ""and is the starting point for ports to all other platforms."]

cv = CountVectorizer()

# fit:特征提取  transform: 将提取的特征转化为词频矩阵
cv.fit(data)

feature_name = cv.get_feature_names()
target = cv.transform(data)

# 将词频矩阵转化为稀疏矩阵
target_toarray = target.toarray()  # shape (len(data), len(feature_name))

data_inverse = cv.inverse_transform(target)

print("feature_name:", feature_name,"\ntarget_toarray:", target_toarray, "\ndata_inverse:", data_inverse)
