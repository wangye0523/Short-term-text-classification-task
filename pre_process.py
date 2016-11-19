#!/usr/lib/env python
#-*- coding=utf-8 -*-

import jieba
import jieba.analyse
import xlrd
import numpy as np
import json
import os
import re
import codecs
# jieba.load_userdict("key_word.txt")
jieba.analyse.set_stop_words("stopwords.txt")

path= os.getcwd()
files=os.listdir(path)
xls_file=[]
for file in files:
    if file.endswith('.xls'):
        xls_file.append(file)
#         print file
# s=re.findall(file,files)




label_com = []
comp_name = []
# key_word =[]
#
for xls in xls_file:
    label_test = re.match('(.+)_(.+).xls',xls)

    # print type(xls),(label_test.group(1)),label_test.group(2)
    label_1 = label_test.group(1).decode('utf-8')
    label_2 = label_test.group(2).decode('utf-8')
    # key_word.append(label_1)
    # key_word.append(label_2)
    data  = xlrd.open_workbook(xls)
    table = data.sheet_by_name(u'公司公告')
    # ncols = table.ncols
    # nrows = table.nrows
    s=table.col_values(2)
    s=s[1:-2]
    # colname = table.row_value(2)
    i=0
    for ss in s:
        # try:
        #     comp_name_tmp = re.match('(.+):(.+)',ss)
        #     comp_name_tmp_1 = comp_name_tmp.group(1)
        #     comp_name.append(comp_name_tmp_1)
        # except:
        #     print "%d th the exception: % s " % i, % ss
        #     i=i+1
        #
        ss= [ss,label_1,label_2]
        label_com.append(ss)
# key_word = set(key_word)
# with open('key_word.txt','w') as f1:
#     f1.write(str(key_word))
# exit()
label_comp=[]
for ii1 in range(0,len(label_com),7):
    label_comp.append(label_com[ii1])


# with open('company_name.txt', 'w') as fff1:
#     # json.dump(comp_name,fff1,indent=4, ensure_ascii=False)
#     fff1.write(str(comp_name))
# exit()
all_sentence=[]
corpus= []
first_label=[]
second_label = []
all_extractag=[]
# f=f[0:5]
for ff in label_comp:
    # print ff[0]
    # seg_list =[]
    seg_list = jieba.lcut(ff[0], cut_all=False)  # Use Precise mode to cut the word
    tttt_11 = jieba.analyse.extract_tags(''.join(seg_list), topK=10)
    # all_sentence.append(','.join(tttt_11))
    all_sentence.append(''.join(seg_list))
    all_extractag.append(tttt_11)
    first_label.append(ff[1])
    second_label.append(ff[2])


# for ii1 in range(len(given_documents)):
#     seg_list = jieba.lcut(given_documents[ii1], cut_all=False)
#     tttt_11 = jieba.analyse.extract_tags(''.join(seg_list), topK=30)
#
#     all.append(seg_list)
#     all_extractag.append(tttt_11)

import pickle

with open('data.pickle', 'w') as fff:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(all_sentence, fff, pickle.HIGHEST_PROTOCOL)
    # pickle.dump(all_extractag, fff, pickle.HIGHEST_PROTOCOL)
    pickle.dump(first_label, fff, pickle.HIGHEST_PROTOCOL)
    pickle.dump(second_label, fff, pickle.HIGHEST_PROTOCOL)


print '*****'*40
