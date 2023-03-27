# coding=utf8
import pandas as pd
import pydotplus as pydotplus
from sklearn.datasets import load_iris  # 导入方法类
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from IPython.display import Image
import os

os.environ["PATH"] += os.pathsep + 'E:\graphviz\Graphviz 2.44.1\\bin'



column = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)']
iris = load_iris()  # 导入数据集iris
iris_feature = iris.data  # 特征数据
iris_target = iris.target  # 分类数据、
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# 输出数据集
df.columns = column
df.hist()
plt.show()

#预剪枝

clf = DecisionTreeClassifier(max_depth=3, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=420)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print("精度是:{:.3f}".format(clf.score(x_test, y_test)))

from six import StringIO
import pydot
#需要安装pydot包，用Anaconda Prompt安装，需要先安装graphviz再安装pydot，命令如下：
# conda install graphviz
# conda install pydot
dot_data = StringIO()
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_pdf("iris.pdf")
