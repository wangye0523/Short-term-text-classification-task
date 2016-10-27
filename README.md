# project--短文本分类
###update by OCT 27
####服务器内存条还没到，还是用的1/10的数据集，试了SVC first 71% second 85% ,KNN first 71% second 78% by CountVectorizer, LR first 72% second 85%. PS:切记数据量大了一定要用LinearSVM而不是SVM。速度可能快了1000倍不止。


###update by OCT.22
####买错了服务器内存条（原来服务器内存条不能mix brand，而且买错了ecc register，好吧我是猪），还是只能用1/10的数据集做实验。试了ti-idf提feature，而不再是之前利用全文本，准确率有提高到78%。我认为不明显。可能还是因为文本短。会继续尝试换分类器。



###update by OCT.13 
####一共有40W条短文本，需要按大标题分8类，小标题分59类。都是supervised training。feature是用用的CountVectorizer（会再试试其他tfidf之类的），分类器暂时只用了MultinomialNB，准确率大概是68%和73%。等机器升了级，还会陆续尝试NN和SVM等主流分类器。


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
