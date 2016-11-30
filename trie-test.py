#!/usr/lib/env python
#-*- coding=utf-8 -*-

import datrie
import pickle
import numpy as np
import gensim
#
# with open('data.pickle', 'rb') as fff:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     all_sentence= np.array(pickle.load(fff))
#     all_extractag= np.array(pickle.load(fff))
#     first_label = pickle.load(fff)
#     second_label= pickle.load(fff)
#
# from gensim import corpora, models
# ###TF-IDF
# dictionary = corpora.Dictionary(all_extractag,prune_at = 2000000)
# corpus = [dictionary.doc2bow(text) for text in all_extractag]
# # doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示
#
# #  topic model, TF-IDF
# tfidf = models.TfidfModel(corpus)
# corpus_tfidf = tfidf[corpus]

score_1_first_NB= [ 0.70121951 , 0.65276253 , 0.6756526  , 0.66268147 , 0.64657066]
print '-'*40