## 功能

中文新闻文本多分类 (utf-8 encoding)

用法：

    ./do_clf.py <input_file> <seg_field> <out_file>

模型文件放在目录./model, 输入文件默认分词，分词格式为hello@nx||world@nx ...

## 训练新模型

如果需要从头训练新的分类模型，也是可以的，分为两步

1. 训练tf_idf模型

    ./doc_reader.py
    
    默认输入文件是./raw_data/day7/uniq，输出文件有多个：
    
        ./model/model.tfidf.dat
        ./model/model.tfidf.dat.dict
        ./model/model.tfidf.dat.dict.2

2. 训练分类模型

    ./do_train.py
    
    默认输入文件是./raw_data/train_val.txt, 输出是./model/clf.final.pkl
    
    当前效果:

    train_accuracy: 0.993

    test_accuracy: 0.860

## 模型优化思路

1. 添加停用词，挖掘高频词筛选

2. 使用卡方测试筛选特征 (see ./doc_reader.py)

3. co-training 扩充语料 (see co_training.py)

    不同分类器, 不同字段(标题或正文)

## 依赖

+ pip install numpy
+ pip install scipy
+ pip install sklearn
+ pip install gensim
+ pip install matplotlib (need _tkinter)

## 数据

+ 训练数据 ./data/train.txt
+ 验证数据 ./data/val.txt
+ 测试数据 ./data/test.txt

如果有新的数据，可以用脚本./data/split.sh分割

数据格式:

    1 label
    2 标题md5
    3 正文md5
    4 标题simhash
    5 内容simhash
    6 网页指纹
    7 标题
    8 内容
    9 时间
    10 媒体名
    11 企鹅号
    12 公众号
    13 文章url
    14 图片url
    15 渠道
    16 类别
    17 载体
    18 预留
    19 tag
    20 阅读数
    21 点赞数
    22 停留数
    23 评论停留时长
    24 标题分词
    25 正文分词
    26 标题正文分词
