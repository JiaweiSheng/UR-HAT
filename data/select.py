# import torch
# from transformers import BertTokenizer
import xml.etree.ElementTree as ET
# import numpy as np
import os
import pickle
import re
import pdb
from collections import Counter, OrderedDict
from math import log2

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif


class Data():
    # def __init__(self, data_path, saved_data_path, data_idx, label2idx, cv_index=0, is_training=True, config=None):
    def __init__(self, data_path, language='en'):
        self.document_data = None
        self.data = []
        self.document_max_length = 512
        self.sentence_max_length = 150

        # if True:
        self.document_data = []
        count = 0
        tree = ET.parse(data_path)  # 根结点
        root = tree.getroot()
        sentence_list_all = []
        label_all = []

        for doc in root[0]:
            id = doc.attrib['id']  # 文档id
            # label = label2idx[doc.attrib['document_level_value']]  # 文档事实标签
            trigger_word_list = []
            flag = False
            sentence_list = []

            sent_one_n = 0
            # 处理句子
            for sent in doc:
                # 句子空
                if sent.text == '-EOP- .' or sent.text == '.':
                    continue
                # 句子非空，重新获得句子，结尾句号
                s = ''
                for t in sent.itertext():  # 获取句子的全部文本
                    s += t
                s = s.replace('-EOP- .', '.').lower()

                # 句子为日期，跳过
                # if re.match(r'\d{4}\D\d{2}\D\d{2}\D\d{2}:\d{2}\D$', s) is not None:  # 日期
                if re.match(r'\d{4}\D\d{2}\D\d{2}\D\d{2}:\d{2}\D\D$', s) is not None:  # 日期
                    flag = True
                    continue
                elif flag:
                    flag = False
                    if len(sent) == 0:  # 句子长度
                        continue
                # 句子过短，跳过
                if len(s) <= 4:
                    continue

                if len(sent) > 0:  # 存在事件
                    # pdb.set_trace()
                    tmp = sent.text.lower() if sent.text is not None else ''
                    # 如果存在 事件
                    for event in sent:
                        pass

                # sent_one_n += 1
                # if len(sent)>1:
                #     print(len(sent))
                #     # print(sent.text)
                #     for event in sent:
                #         # print(event.text, event.attrib['sentence_level_value'])
                #         pass
                if len(sent) > 0:
                    # has triggers
                    tmp = sent.text.lower() if sent.text is not None else ''
                    for event in sent:
                        l = event.attrib['sentence_level_value']
                        # print(s,l, event.text)
                        sentence_list_all.append(s.split())
                        label_all.append(l)

        self.sentence_list_all, self.label_all = sentence_list_all, label_all


def gen_vocab(sentences_all):
    vocab = {}
    for sent in sentences_all:
        for word in sent:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    # 去除低频词
    vocab_filted = {}
    for w in vocab:
        if vocab[w] > 3:
            vocab_filted[w] = vocab[w]
    return vocab_filted


def frac_score(count_type_word, count_type, count_word):
    scores = {}
    for w in count_word:
        if w in count_type_word and w in count_word:
            scores[w] = count_type_word[w] / (count_type * count_word[w])
    scores_sorted = sorted(scores.items(), key=lambda k: k[1], reverse=True)
    print(scores_sorted[:100])
    return scores


def select_key_words(sentences_all, labels_all, vocab):
    label_set = set(labels_all)
    l2s_dict = {}
    for l in label_set:
        l2s_dict[l] = []
    for i, sent in enumerate(sentences_all):
        label = labels_all[i]
        l2s_dict[label].append(sent)

    count_word = vocab
    for l in label_set:
        print(l, len(l2s_dict[l]))
        count_type = len(l2s_dict[l])
        count_type_word = gen_vocab(l2s_dict[l])
        print('句子中的词数', len(count_type_word))
        # vocab_sorted = sorted(count_type_word.items(),key=lambda k:k[1], reverse=True)
        # print(vocab_sorted[:10])
        # pdb.set_trace()
        scores = frac_score(count_type_word, count_type, count_word)


def multual_infomation(N_10, N_11, N_00, N_01):
    """
    互信息计算
    :param N_10:
    :param N_11:
    :param N_00:
    :param N_01:
    :return: 词项t互信息值
    """
    N = N_11 + N_10 + N_01 + N_00
    I_UC = (N_11 * 1.0 / N) * log2((N_11 * N * 1.0) / ((N_11 + N_10) * (N_11 + N_01))) + \
           (N_01 * 1.0 / N) * log2((N_01 * N * 1.0) / ((N_01 + N_00) * (N_01 + N_11))) + \
           (N_10 * 1.0 / N) * log2((N_10 * N * 1.0) / ((N_10 + N_11) * (N_10 + N_00))) + \
           (N_00 * 1.0 / N) * log2((N_00 * N * 1.0) / ((N_00 + N_10) * (N_00 + N_01)))
    return I_UC


label_vocab = {'Uu': 0, 'CT+': 1, 'PS+': 2, 'PS-': 3, 'CT-': 4}


def text_classify(sentences, labels):
    pdb.set_trace()
    y_labels = [label_vocab[l] for l in labels]
    count_vec = CountVectorizer(binary=True)
    # doc_train_bool = count_vec.fit_transform(sentences)
    # doc_test_bool = count_vec.transform(doc_terms_test)
    X_sent = count_vec.fit_transform(sentences)
    mi_scores = mutual_info_classif(X_sent, y_labels)


d = Data(data_path='./data/english.xml')
# d = Data(data_path='./data/chinese.xml', language='zh')
sentences_all, labels_all = d.sentence_list_all, d.label_all
# vocab = gen_vocab(sentences_all)
text_classify(sentences_all, labels_all)

# d = Data(data_path='./data/english.xml')
# # d = Data(data_path='./data/chinese.xml', language='zh')
# sentences_all, labels_all = d.sentence_list_all, d.label_all
# vocab = gen_vocab(sentences_all)
# vocab_sorted = sorted(vocab.items(), key=lambda k: k[1], reverse=True)
# print(vocab_sorted[:10])
# select_key_words(sentences_all, labels_all, vocab)
# # pdb.set_trace()
# print(len(vocab_sorted))
'''
<sentence id="CS3921.10">同一天，原本宣称要<event id="CE3921.0" sentence_level_value="CT+">冻结</event>对土军售交付的德国口风出现松动，总理安格拉·默克尔表示，不会全部<event id="CE3921.1" sentence_level_value="CT-">冻结</event>。</sentence>
'''