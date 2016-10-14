# project--短文本分类

##update by OCT.13 
###一共有40W条短文本，需要按大标题分8类，小标题分59类。都是supervised training。feature是用用的CountVectorizer（会再试试其他tfidf之类的），分类器暂时只用了MultinomialNB，准确率大概是68%和73%。等机器升了级，还会陆续尝试NN和SVM等主流分类器。


###update by Oct.7
####将分好词的短文本做向量化，用的CountVectorizer。分类器用的MultinomialNB。把数据集缩小了10倍，每10条连续的短文本里取出一条。交叉验证为7folders，first_label大约68%准确率，second_label为73%。


###update by Oct.5
####因为电脑性能原因，全部跑要down，经过测试，只有当把数据集缩小10倍后才能跑得起。

###update by Sep.25
####统计信息，first_label 有8类标题， second_label有59类标题。总计409871条短文本，分出了5349348个词语


###update by Sep.20
####处理了所有xls文件，清洗格式，写在label.jason里面。 

##数据的设置在 pre_process.py 的 46行
##RUN process.py 即可
