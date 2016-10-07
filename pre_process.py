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
#
#
# exit()
for xls in xls_file:
    label_test = re.match('(.+)_(.+).xls',xls)

    # print type(xls),(label_test.group(1)),label_test.group(2)
    label_1 = label_test.group(1).decode('utf-8')
    label_2 = label_test.group(2).decode('utf-8')

    data  = xlrd.open_workbook(xls)
    table = data.sheet_by_name(u'公司公告')
    # ncols = table.ncols
    # nrows = table.nrows
    s=table.col_values(2)
    s=s[1:-2]
    # colname = table.row_value(2)
    i=0
    for ss in s:
        ss= [ss,label_1,label_2]
        label_com.append(ss)
label_comp=[]
for ii1 in range(0,len(label_com),10):
    label_comp.append(label_com[ii1])
# label_com=label_com[0:50000]

# with codecs.open('label.json', 'w', encoding='utf-8') as fff1:
#     json.dump(label_com,fff1,indent=4, ensure_ascii=False)


all_sentence=[]
corpus= []
first_label=[]
second_label = []
# f=f[0:5]
for ff in label_comp:
    # print ff[0]
    # seg_list =[]
    seg_list = jieba.lcut(ff[0], cut_all=False)  # Use Precise mode to cut the word
    all_sentence.append(','.join(seg_list))
    first_label.append(ff[1])
    second_label.append(ff[2])




import pickle

with open('data.pickle', 'w') as fff:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(all_sentence, fff, pickle.HIGHEST_PROTOCOL)
    pickle.dump(first_label, fff, pickle.HIGHEST_PROTOCOL)
    pickle.dump(second_label, fff, pickle.HIGHEST_PROTOCOL)


print '****'*40