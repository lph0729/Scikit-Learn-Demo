#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-7-17 下午2:59
@email: lph0729@163.com  

"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import warnings


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 设置标记点和颜色
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 绘制决策面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # 高粱所有的数据点
    if test_idx:
        # 绘制所有数据点
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')


if __name__ == '__main__':
    iris = load_iris()

    x = iris.data[:, [2, 3]]
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    lr = LogisticRegression(C=10000.0, random_state=0)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    lr_score = lr.score(x_test, y_test)

    report = classification_report(y_true=y_test, y_pred=y_pred)

    # # 绘制决策边界
    x_combine = np.vstack((x_train, x_test))
    y_comnine = np.hstack((y_train, y_test))
    plot_decision_regions(x_combine, y_comnine,
                          classifier=lr, test_idx=range(105, 150))

    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")

    plt.legend(loc="upper left")
    plt.savefig("./iris.png")
    plt.show()

    print("y_test:\n", y_test[:10], "\ny_pred:\n", y_pred[:10], "\nscore:\n", lr_score, "\nreport:\n", report)
