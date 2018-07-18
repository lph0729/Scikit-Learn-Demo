# Scikit-Learn-Demo
主要是通过sklearn数据集的学习来了解机器学习过程

1.环境介绍：

    运行环境： Linux + Python3.5 + Sklearn

    创建虚拟环境： mkvirtualenvs sklearn_envs

    安装包: pip install sklearn  pip install scipy

2.机器学习常用算法：

    1).监督学习：

        分类算法： K-近邻算法、 朴素贝叶斯、 决策树与随机森林、 逻辑回归、 神经网络

        回归算法： 线性回归、 岭回归、 Lasso回归

    2).无监督学习：

        聚类算法： k-means

3.特征工程：最大限度的从原始数据中提取特征以供算法和模型使用

    1).数据的特征抽取： 就是从原始数据中通过计算得到一些特征

        (1).英文文本的特征抽取: CountVectorizer

        (2).字典特征的抽取: DictVectorizer

        (3).中文文本特征的抽取: CountVectorizer

        (4).TF-IDF特征抽取: TfidfVectorizer

    2).数据的特征处理：

        (1).归一化：将特征值转化为[0, 1]区间

            公式： X = (x - min)/ (max - min)

            接口： MinMaxScaler

            属性： min、 max、 average

            缺点： 最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景

        (2).标准值化： 把原始数据变换到均值为0(转化之后每一列的数据平均值为0),方差为1范围内

            公式： X = (x - mean)/ σ   σ = std^1/2   mean: 每一列数据的平均值

            接口：StandardScaler

            属性： mean、 std

            场景：在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景

        (3).缺失值: 处理方式

            删除： 如果每列或者行数据缺失值达到一定的比例，建议放弃整行或者整列

            插补： 可以通过缺失值每行或者每列的平均值、中位数来填充


        (4).特征选择: 从提取到的所有特征中选择部分特征作为训练集特征，特征在选择前和选择后可以改变值、也不改变值. 主要方法（三大武器）

            特征选择原因： 1). 去冗余  2).降噪声

            Filter(过滤式): VarianceThreshold

            Embedded(嵌入式)：正则化、决策树

            Wrapper(包裹式)

        (5).降维: 将原始高维属性空间转变为一个低维子空间

            主成成分分析(principal component analysis,PCA): 在PCA中，数据从原来的坐标系转换到新的坐标系

            因子分析(Factor Analysis)

            独立成分分析(Independent Component Analysis，ICA)

4.分类算法的介绍：

    1).KNN: 如果一个样本在数据集中，有k个最相似的样本，而k个样本大多数属于某一个类别，那么这个样本也属于该类别

        原理解释： 1）计算测试数据与各个训练数据之间的距离；

                  2）按照距离的递增关系进行排序；

                  3）选取距离最小的K个点；

                  4）确定前K个点所在类别的出现频率；

                  5）返回前K个点中出现频率最高的类别作为测试数据的预测分类

        公式： 欧拉公式

        接口： KNeighborsClassifier

        超参数： n_neighbors

        优缺点：

            优点： 简单，易于理解，易于实现

            缺点： 懒惰算法，对测试样本分类时的计算量大，内存开销大； 必须指定K值，K值选择不当则分类精度不能保证

        适用场景： 小数据场景，几千～几万样本，具体场景具体业务去测试

    2).交叉验证： 在网格搜索中每组超参数都采用交叉验证来进行评估。

        接口： GridSearchCV

        参数： estimator、 param_grid(估计器参数:dict)、 cv(指定几折交叉验证)

        结果属性： best_score_、 best_estimator_、 cv_results_

    3).朴素贝叶斯算法：

        概率论基础： 联合概率 P(A、B)、 条件概率 P(B|A)

        贝叶斯准则： 基于条件概率以及公式P(A∩B=P(A).P(B|A)=P(B).P(A|B))推导

            公式： P(Ci|x,y) = (P(Ci).P(x, y|Ci)) / P(x, y)

            P(Ci): 分类ci的概率

            P(x, y|Ci): 在分类为Ci的条件下特征值为x, y的概率

            P(x, y): x, y的联合概率

        朴素贝叶斯准则： 当特征值x，y相互独立时，P(x,y|ci)=P(x|Ci)P(y|Ci), 则

            公式： P(Ci|x,y) = (P(Ci).P(x|Ci).P(y|Ci)) / P(x, y)

        接口： MultinomialNB

        优缺点：

            优点：　有稳定的分类效率；　常用于文本分类；　分类准确率高，速度快

            缺点：　基于假设每个特征是相互独立的，某些时候会由于假设的先验模型的原因导致预测效果不佳

    4).决策树(Decision Tree)：

        算法：　ID3算法表示 信息熵来表示信息的不确定性

        信息熵： H(x) = -ΣP(Xi).logP(Xi)

            信息熵是代表随机变量的不确定度，信息熵越大，不确定性越大

        条件熵：H(Y|X) = ΣP(Xi).H(Y|X=xi)  条件熵小于信息熵

        信息增益 = H(x) - H(Y|X) 信息增益代表了在一定程度下，信息不确定性减少的程度

        接口： DecisionTreeClassifier(criterion=’gini’, max_depth=None,random_state=None)

        参数：criterion: 默认是’gini’系数，也可以选择信息增益的熵’entropy'

            max_depth: 树的最大深度。如果为None，则扩展节点直到所有叶子都是纯的或直到所有叶子包含少于min_samples_split样本

            decision_path: 返回决策树的路径

        保存dot文件： export_graphviz(decision_tree=dtc, out_file="./tree.dot", feature_names=None)

        展示决策树：graphviz 能够将dot文件转换为pdf、png

            安装graphviz： apt-get install graphviz

            运行： dot -Tpng tree.dot -o tree.png

    5).随机森林算法(Random Forest):

        1.集成学习介绍：

            (1).随机森林属于集成学习（Ensemble Learning）中的bagging算法。在集成学习中，主要分为bagging算法和boosting算法

            (2).bagging的算法过程如下：

                从原始样本集中使用Bootstraping方法随机抽取n个训练样本，共进行k轮抽取，得到k个训练集。（k个训练集之间相互独立，元素可以有重复）

                对于k个训练集，我们训练k个模型（这k个模型可以根据具体问题而定，比如决策树，knn等）

                对于分类问题：由投票表决产生分类结果；对于回归问题：由k个模型预测结果的均值作为最后预测结果。（所有模型的重要性相同）

            (3).Bagging与Boosting的主要区别

                样本选择上：Bagging采用的是Bootstrap随机有放回抽样；而Boosting每一轮的训练集是不变的，改变的只是每一个样本的权重。

                样本权重：Bagging使用的是均匀取样，每个样本权重相等；Boosting根据错误率调整样本权重，错误率越大的样本权重越大。

                预测函数：Bagging所有的预测函数的权重相等；Boosting中误差越小的预测函数其权重越大。

                并行计算：Bagging各个预测函数可以并行生成；Boosting各个预测函数必须按顺序迭代生成。

            (4).下面是将决策树与这些算法框架进行结合所得到的新的算法：

                Bagging + 决策树 = 随机森林

                AdaBoost + 决策树 = 提升树

                Gradient Boosting + 决策树 = GBDT

        2.决策树算法原理：

            从原始训练集中使用Bootstraping方法随机有放回采样选出m个样本，共进行n_tree次采样，生成n_tree个训练集

            对于n_tree个训练集，我们分别训练n_tree个决策树模型

            对于单个决策树模型，假设训练样本特征的个数为n，那么每次分裂时根据信息增益/信息增益比/基尼指数选择最好的特征进行分裂

            每棵树都一直这样分裂下去，直到该节点的所有训练样例都属于同一类。在决策树的分裂过程中不需要剪枝

            将生成的多棵决策树组成随机森林。对于分类问题，按多棵树分类器投票决定最终分类结果；对于回归问题，由多棵树预测值的均值决定最终预测结果

        3.接口函数： RandomForestClassifier(n_estimators=10, criterion=’gini’,max_depth=None, bootstrap=True, random_state=None)

        4.参数：

            n_estimators: 森林里树的数量(default = 10)

            criterion: 算法类别,默认是“gini”

            bootstrap: 是否在构建树时使用放回抽样(default=True)

        5.决策树优缺点：

            优点： 能够解决单个决策树不稳定的情况; 能够处理具有高维特征的输入样本，而且不需要降维(使用的是特征子集); 对于缺省值问题也能够获得很好得结果（投票）

            缺点： 随机森林已经被证明在某些噪音较大的分类或回归问题上会过拟

    6).分类模型的评估

        准确率：accurcay = (TP+FN)/(TP+TN+FP+FN)

        精确率： 2precision = (TP)/(TP+FP)

        召回率： recall = (TP)/(TP+FN)

        F1-score: 反映了模型的稳健型

            F1-score = 2TP/(2TP+FP+FN)=2precision.recall/precision+recall

5.回归算法的介绍：

    1).线性回归： 求线性回归方程系数的过程就是回归。 线性回归的目标就是找到回归系数，构建回归方程(模型)来预测目标值

        两种线性回归算法：

            (1).正规方程(Normal Equation)： 普通最小二乘法   接口 LinearRegression

            (2).梯度下降(Gradient Descent):  接口 SGDRegressor

            (3).两种算法的对比：

                     梯度下降                           正规方程

                  需要选择学习率                        不需要学习率

                  需要多次迭代                           一次求解

               当特征n很大时，也能较好使用          需要计算(X^TX)^-1,当特征值很大时，计算代价较大

                  适用于各种模型                   只适用于线性模型，不适合逻辑回归等其他模型

        目标函数： loss_func = Σ(y - y_pred)^2 = Σ(y - X^TW)^2

        回归性能评估： 均方误差函数(mse)  mse = Σ(Yi - (Y_pred)mean)^2

            接口： mean_squared_error(y_true, y_pred)

            参数： 真实值，预测值为标准化之前的值

    2).两种正则化算法：

        欠拟合与过拟合：

            欠拟合解决： 在训练集以及训练集之外的数据集也不能很好的拟合,此时就认为出现了欠拟合

                原因： 学习到的特征过少

                解决方案： 增加数据集的特征数量

            过拟合解决： 训练集数据能够很好的拟合，但是训练集之外的数据集不能很好的拟合

                原因： 原始特征过多,存在一些嘈杂特征,模型过于复杂是因为模型尝试兼顾各个测试数据点

                解决方案： (1).进行特征选择,消除关联性大的特征(很难做)

                         (2).交叉验证(让所有数据都有过训练)

                         (3).正则化

        正则化：减少特征变量的数量级

            L2正则化： L2正则化就是在损失函数后面添加一个L2正则化项

                H(w) = loss_func = Σ(y - y_pred)^2 = Σ(y - X^TW)^2

                正则化： min{H(w) + λ/2n.Σ w^2} 被L2正则化的线性回归称为岭回归

            L1正则化： L1正则化就是在损失函数后面添加一个L1正则化项

                H(w) = loss_func = Σ(y - y_pred)^2 = Σ(y - X^TW)^2

                正则化： min{H(w) + λ/2m.Σ|w|} 被L1正则化的线性回归称为Lasso回归

            L1与L2之间的区别：

                L1可以出现回归系数缩小至0的情况，可以用来进行特征筛选

                L2不会出现回归系数为0的情况，也就是保存所有特征

        接口：

            L1接口： Ridge    RidgeCV

            L2接口： Lasso    LassoCV

    3).逻辑回归： 主要是解决二分类问题

        逻辑回归：简单的理解就是线性回归 + 阀值(阀值是通过sigmoid函数来确定)将范围转化为(0, 1)来完成分类

        逻辑回归公式： Y = g(z) = g(WX^T)

                     g(z) = 1/(1 + e^-1)

        目标函数： loss_func = -Y x logY预测 + (Y - 1) x log(1 - Y预测)

        优缺点：

            优点： 适合需要得到一个分类概率的场景

            缺点： 当特征空间很大时，逻辑回归的性能不是很好（看硬件能力）

6.非监督学习：

    1).K-Means聚类算法

        聚类介绍： 聚类算法是一种典型的无监督学习算法，主要用于将相似的样本自动归到一个类别中，聚类算法与分类算法最大的区别是：聚类算法是无

            监督的学习算法，而分类算法属于监督的学习算法。在聚类算法中根据样本之间的相似性，将样本划分到不同的类别中，对于不同的相似度计算

            方法，会得到不同的聚类结果，常用的相似度计算方法有欧式距离法

        K-Means： 因为它可以发现K个不同的簇，每个簇的中心采用簇中所含值的均值计算而成

        K-Means算法计算过程：

            (1).随机设置K个点作为初始的聚

            (2).重复计算一下过程，直到质心不再改变

                计算其他每个点到K个中心的距离，每个点与离他最近的聚类中心归为一个簇，形成K个簇

                重新计算出每个聚类的新中心点（平均值），如果新的中心点与原来K个中心一样，则聚类结束

            (3).输出最终的质心以及每个类

7.代码解读：

    sklearn-feature-engineer: 主要是对数据集的特征处理

        feature_extraction: 主要是对数据集的4中特征抽取

            jieba_cut.py: 使用jieba对中文字符串进行分词

            sklearn_datasets.py: 主要是通过sklearn数据集iris与boston来认识热症处理过程中的变量

            skleearn_english_text_extraction.py: 英文文本特征的提取

            sklearn_dict_extraction.py: 字典特征的提取

            sklearn_chinese_text_extraction.py: 中文文本特征的提取

            sklearn_TF-IDF.py: 词频、逆文档频率来统计特征

        feature_preprocess: 对数据集的5种特征处理

            sklearn_max_min_scaler.py: 特征的归一化处理

            sklearn_standardscler.py: 特征的标准化处理

            sklearn_imputer.py: 特征缺失值处理

            sklearn_feature_choice.py: 特征选择

            sklearn_pca.py: 使用pca对特征进行降维处理

    sklearn-model-training: 主要是对已有的数据进行模型训练以及预测

        classification: 分类算法的模型训练

            sklearn_datasets_split.py: 对鸢尾花数据集进行训练集与测试集划分

            sklearn_knn.py: 使用knn算法预测提供的数据集分类

            sklearn_knn_iris.py: 使用knn算法对鸢尾花数据集进行分类

            sklearn_cross_validation.py: 使用网络交叉实现对入住位置的预测

            skearn_naive_bayes.py: 使用朴素贝叶斯对新闻进行分类

            sklearn_decision_tree.py: 使用决策树对泰坦尼克号游客生还率进行预测

            sklearn_random_forest.py: 使用随机森林对泰坦尼克号游客生还率进行预测

        regression: 回归算法的魔性训练

            sklearn_linear_model.py: 分别使用正规方程、梯度下降算法、岭回归、Lasso回归算法对boston房价进行预测对比

            sklearn_logistic_model.py: 使用逻辑回归对iris数据集进行分类

            sklearn_k-means.py: 使用k-means算法对iris数据集进行聚类分类



















