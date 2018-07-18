#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-16 下午4:54
@email: lph0729@163.com  

"""
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error

boston = load_boston()

feature_data = boston.data
target_data = boston.target

x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data)
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))

# 线性回归模型预测模型
lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_pred = lr.predict(x_test)

lr_error = mean_squared_error(y_pred=ss_y.inverse_transform(y_lr_pred), y_true=y_test)

# 梯度下降预测模型
sgd = SGDRegressor()
sgd.fit(x_train, y_train)

y_sgd_pred = sgd.predict(x_test)

sgd_error = mean_squared_error(y_true=y_test, y_pred=ss_y.inverse_transform(y_sgd_pred))

# 岭回归模型预测
ridge = Ridge(alpha=3)
ridge.fit(x_train, y_train)

y_ridge_pred = ridge.predict(x_test)

ridge_error = mean_squared_error(y_pred=ss_y.inverse_transform(y_ridge_pred), y_true=y_test)

# Lasso回归
lasso = Lasso(alpha=0.01)
lasso.fit(x_train, y_train)

y_lasso_pred = lasso.predict(x_test)

lasso_error = mean_squared_error(y_pred=y_lasso_pred, y_true=y_test)

alphas = [0.01, 0.1, 1, 5, 10, 20, 50, 100]
# 使用交叉岭回归预测模型
rig_cv = RidgeCV(alphas=alphas)
rig_cv.fit(x_train, y_train)

# 使用交叉验证lasso回归进行模型预测
lasso_cv = LassoCV(alphas=alphas)
lasso_cv.fit(x_train, y_train)


print("feature_data:\n", feature_data[:10], "\ntarget:\n", target_data[:10], "\n正规方程的均方误差为:\n", lr_error,
      "\n正规方程的回归系数:\n", lr.coef_, "\n梯度下降的均方误差:\n", sgd_error, "\n梯度下降的回归系数:\n", sgd.coef_,
      "\n岭回归方程误差:\n", ridge_error, "\n岭回归均系数：\n", ridge.coef_, "\nlasso回归方程误差:\n", lasso_error,
      "\nlasso回归均方系数:\n", lasso.coef_, "\n岭回归最优的正则化力度:\n", rig_cv.alpha_, "\nlasso回归最优的正则化力度:\n",
      lasso_cv.alpha_)
