#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-13 上午9:06
@email: lph0729@163.com  

"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

news = fetch_20newsgroups(subset="all", data_home=None)  # data_home:None时，默认下载到~/scikit_learn_data文件下
x = news.data
y = news.target

x_train, x_test, y_train, y_test = train_test_split(x, y)

tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test).toarray()
feature_name = tfidf.get_feature_names()

mnb = MultinomialNB(alpha=1.0)  # 拉普拉斯平滑系数,默认1
mnb.fit(x_train, y_train)

y_pred = mnb.predict(x_test)
scores = mnb.score(x_test, y_test)

print("x:\n", x[0], "\ny:\n", y[:10], "\ny_pred:\n", y_pred[:10], "\nfeature_name:\n", feature_name, "\nscores:\n",
      scores)
