#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-11 下午2:29
@email: lph0729@163.com  

"""
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

str_1 = "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天"

str_2 = "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去"

str_3 = "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系"

tf_idf = TfidfVectorizer()

data = [str_1, str_2, str_3]
content = [" ".join(jieba.cut(sentence)) for sentence in data]


result = tf_idf.fit_transform(content)
feature_names = tf_idf.get_feature_names()
target = result.toarray()
content_inverse = tf_idf.inverse_transform(result)
print("feature_name:", feature_names, "\ntarget:", target, "\ncontent_inverse:", content_inverse)