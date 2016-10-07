#!/usr/lib/env python
#-*- coding=utf-8 -*-

import jieba
import jieba.analyse
import numpy as np
import pickle
jieba.analyse.set_stop_words("stopwords.txt")


with open('data.pickle', 'rb') as fff:
    # Pickle the 'data' dictionary using the highest protocol available.
    all_sentence= np.array(pickle.load(fff))
    first_label = pickle.load(fff)
    second_label= pickle.load(fff)



from sklearn import preprocessing
label_preprocess_1  = preprocessing.LabelEncoder()
first_array         = label_preprocess_1.fit(first_label)
##################

label_preprocess_2  = preprocessing.LabelEncoder()
second_array        = label_preprocess_2.fit(second_label)

########


from sklearn.feature_extraction.text import CountVectorizer

####
vectorizer = CountVectorizer()
all_sentence_tmp = vectorizer.fit_transform(all_sentence)
all_sentence_vec = all_sentence_tmp.toarray()


# transformer = TfidfTransformer()
# tfidf       = transformer.fit_transform(vectorizer.fit_transform(all_sentence_vec))

from sklearn import cross_validation


#########################
from sklearn.naive_bayes import MultinomialNB
MNB_model = MultinomialNB(alpha=0.01)
scores1 = cross_validation.cross_val_score(MNB_model, all_sentence_vec, first_label, cv=5)
scores2 = cross_validation.cross_val_score(MNB_model, all_sentence_vec, second_label, cv=5)

########################
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()
scores1 = cross_validation.cross_val_score(LR_model, all_sentence_vec, first_label, cv=5)
scores2 = cross_validation.cross_val_score(LR_model, all_sentence_vec, second_label, cv=5)

print scores1,scores2

#######
##KNN
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
knn_model = KNeighborsClassifier()
scores1 = cross_validation.cross_val_score(knn_model, all_sentence_vec, first_label, cv=5)
scores2 = cross_validation.cross_val_score(knn_model, all_sentence_vec, second_label, cv=5)




exit()




#
# from sklearn.pipeline import Pipeline
# #nbc means naive bayes classifier
# nbc_1 = Pipeline([
#     ('vect', CountVectorizer()),
#     ('clf', MultinomialNB()),
# ])
# nbc_2 = Pipeline([
#     ('vect', HashingVectorizer(non_negative=True)),
#     ('clf', MultinomialNB()),
# ])
# nbc_3 = Pipeline([
#     ('vect', TfidfVectorizer()),
#     ('clf', MultinomialNB()),
# ])
#
# nbcs = [nbc_1, nbc_2, nbc_3]
#
#
#
#
# from sklearn import cross_validation
#
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(all_sentence_vec, first_label_array, test_size=0.3, random_state=0)
#
# # from sklearn import cross_val_score, KFold
# from scipy.stats import sem
#
# def evaluate_cross_validation(clf, X, y, K):
#     # create a k-fold croos validation iterator of k=5 folds
#     cv = cross_validation.KFold(len(y), K, shuffle=True, random_state=0)
#     # by default the score used is the one returned by score method of the estimator (accuracy)
#     scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
#     print scores
#     print ("Mean score: {0:.3f} (+/-{1:.3f})").format(
#         np.mean(scores), sem(scores))
#
# for nbc in nbcs:
#     evaluate_cross_validation(nbc, X_train, Y_train, 5)
#


print '*****'*4
