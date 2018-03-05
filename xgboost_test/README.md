# xgboost文本分类实验

## 运行

训练模型

./do_train.py (test_accuracy: 0.78)

分类文本

 ./do_clf.py ./raw_data/test.txt 25 1.out

## 依赖

+ gensim [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)
+ sklearn [http://scikit-learn.org/](http://scikit-learn.org/)
+ xgboost [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)

## 资料


1. [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/model.html)

	官网教程，阐述算法原理，通俗易懂，里面有个重要PPT [https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)

2. [Awesome XGBoost](https://github.com/dmlc/xgboost/tree/master/demo)

	官网出的例子教程，主要参考这个上手，比如二分类、多类别分类、交叉验证等，还有论文、PPT等其他资料
	
	例子对应github代码目录./xgboost/demo

3. 可调节参数 [https://github.com/dmlc/xgboost/blob/master/doc/parameter.md](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)

	备用，需要时查询

4. [Notes on Parameter Tuning](https://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html)

	官网出的调参指导，比较简单，没啥用

5. [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

	比较实用的调参教程，非常全，可以看第3部分： Parameter Tuning with Example，按照所给6个步骤的来调参。

6. [知乎讨论：GBDT和XGBOOST的区别](https://www.zhihu.com/question/41354392)


## 其他

运行demo中binary classification会出错，修复方法可参考 [https://github.com/dmlc/xgboost/issues/2638](https://github.com/dmlc/xgboost/issues/2638)

