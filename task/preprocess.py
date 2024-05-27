import re
from math import log2
from readline import write_history_file
from typing import Tuple
import xml.etree.ElementTree as ET
import pdb
import jieba
from transformers import BertTokenizer
from collections import Counter
from task.utils import write_jsonl, read_jsonl

import io  
import sys  
 
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码  


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


def chi_square(N_10, N_11, N_00, N_01):
    """
    卡方计算
    :param N_10:
    :param N_11:
    :param N_00:
    :param N_01:
    :return: 词项t卡方值
    """
    fenzi = (N_11 + N_10 + N_01 + N_00) * (N_11 * N_00 - N_10 * N_01) * (N_11 * N_00 - N_10 * N_01)
    fenmu = (N_11 + N_01) * (N_11 + N_10) * (N_10 + N_00) * (N_01 + N_00)
    if fenmu == 0:
        return 0
    return fenzi * 1.0 / fenmu


def freq_select(t_doc_cnt, doc_cnt):
    """
    频率特征计算
    :param t_doc_cnt: 类别c中含有词项t的文档数
    :param doc_cnt: 类别c中文档总数
    :return: 词项t频率特征值
    """
    return t_doc_cnt * 1.0 / doc_cnt


def selectFeatures(documents, category_name, top_k, vocabulary, select_type="chi"):
    """
    特征抽取
    :param documents: 预处理后的文档集
    :param category_name: 类目名称
    :param top_k:  返回的最佳特征数量
    :param select_type: 特征选择的方法，可取值chi,mi,freq，默认为chi
    :return:  最佳特征词序列
    """
    L = []
    # 互信息和卡方特征抽取方法
    if select_type == "chi" or select_type == "mi":
        # pdb.set_trace()
        for t in vocabulary:
            N_11 = 0
            N_10 = 0
            N_01 = 0
            N_00 = 0
            N = 0
            for label, word_set in documents:
                if (t in word_set) and (category_name == label):
                    N_11 += 1
                elif (t in word_set) and (category_name != label):
                    N_10 += 1
                elif (t not in word_set) and (category_name == label):
                    N_01 += 1
                elif (t not in word_set) and (category_name != label):
                    N_00 += 1
                else:
                    print("N error")
                    exit(1)

            if N_00 == 0 or N_01 == 0 or N_10 == 0 or N_11 == 0:
                continue
            # 互信息计算
            if select_type == "mi":
                A_tc = multual_infomation(N_10, N_11, N_00, N_01)
            # 卡方计算
            else:
                A_tc = chi_square(N_10, N_11, N_00, N_01)
            L.append((t, A_tc))
    # 频率特征抽取法
    elif select_type == "freq":
        for t in vocabulary:
            # C类文档集中包含的文档总数
            doc_cnt = 0
            # C类文档集包含词项t的文档数
            t_doc_cnt = 0
            for label, word_set in documents:
                if category_name == label:
                    doc_cnt += 1
                    if t in word_set:
                        t_doc_cnt += 1
            A_tc = freq_select(t_doc_cnt, doc_cnt)
            L.append((t, A_tc))
    else:
        print("error param select_type")
    return sorted(L, key=lambda x: x[1], reverse=True)[:top_k]


class Preprocess():
    def __init__(self):
        pass

    def preprocess(self, data_path, language):
        data = []
        tree = ET.parse(data_path)  # 根结点
        root = tree.getroot()
        for idx, doc in enumerate(root[0]):
            # if idx not in data_idx:  # 用来判定为训练集
            #     continue
            data_one_doc = {}
            id = doc.attrib['id']  # 文档id
            data_one_doc['id'] = id
            value = doc.attrib['document_level_value']  # 文档id
            data_one_doc['value'] = value

            data_one_doc['sentence'] = []

            flag = False
            # 处理句子
            for sent_id, sent in enumerate(doc):
                if language == 'zh':
                    if sent.text == '-EOP-.' or sent.text == '。':
                        continue
                    # 句子非空，重新获得句子，结尾句号
                    s = ''
                    for t in sent.itertext():  # 获取句子的全部文本
                        s += t
                    s = s.replace('-EOP-.', '。').lower()

                    # 句子为日期，跳过
                    if re.match(r'\d{4}\D\d{2}\D\d{2}\D\d{2}:\d{2}\D$', s) is not None:  # 日期
                        flag = True
                        continue
                    elif flag:
                        flag = False
                        if len(sent) == 0:  # 句子长度
                            continue
                    # 句子过短，跳过
                    if len(s) <= 4:
                        continue
                    # pdb.set_trace()
                    words = list(s)
                    text = ''.join(words)
                elif language == 'en':
                    if sent.text == '-EOP- .' or sent.text == '.':
                        continue
                    # 句子非空，重新获得句子，结尾句号
                    s = ''
                    for t in sent.itertext():  # 获取句子的全部文本
                        s += t
                    s = s.replace('-EOP- .', '.').lower()

                    # 句子为日期，跳过
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
                    words = s.split()
                    text = ' '.join(words)
                else:
                    raise NotImplementedError('Unknown language')

                sent_info = {'text': text, 'triggers': [], 'trigger_num': 0}

                if len(sent) > 0:  # 存在事件
                    tmp = sent.text.lower() if sent.text is not None else ''
                    # 如果存在 事件
                    for event in sent:
                        if language == 'en':
                            tmp_subwords = tmp.split()  # 本句子，子词
                            trigger_subwords = event.text.lower().split()  # 触发词
                        elif language == 'zh':
                            tmp_subwords = list(tmp)
                            trigger_subwords = list(event.text.lower())
                        else:
                            raise NotImplementedError('Unknown language')

                        pos0 = len(tmp_subwords)
                        pos1 = pos0 + len(trigger_subwords)

                        if language == 'en':
                            trigger_text = ' '.join(words[pos0:pos1])
                        elif language == 'zh':
                            trigger_text = ''.join(words[pos0:pos1])
                        else:
                            raise NotImplementedError('Unknown language')

                        sent_info['triggers'].append({
                            'sent_id': sent_id,
                            'span': [pos0, pos1],
                            'text': trigger_text,
                            'value': event.attrib['sentence_level_value']
                        })
                        tmp += event.text.lower()
                        if event.tail is not None:  # 标签结束到下一个标签
                            tmp += event.tail.lower()
                        sent_info['trigger_num'] += 1
                data_one_doc['sentence'].append(sent_info)
            data.append(data_one_doc)
        return data

    def conduct(self, data_path, data_path_save, language='en'):
        data = self.preprocess(data_path, language)
        write_jsonl(data_path_save, data)
        data = read_jsonl(data_path_save)
        print('Preprocess finished!')
        print('Example:', data[0])


class DataReader():
    def __init__(self, data, data_idx, language='en'):
        sentence_list_all, label_all = [], []
        # pdb.set_trace()
        for i, doc in enumerate(data):
            if i not in data_idx:
                continue
            for sent in doc['sentence']:
                if sent['trigger_num'] != 0:
                    sent_text = sent['text']
                    words = self.cut_words(sent_text, language)
                    for trigger in sent['triggers']:
                        label = trigger['value']
                        sentence_list_all.append(words)
                        label_all.append(label)
        self.sentence_list_all, self.label_all = sentence_list_all, label_all

    def cut_words(self, sentence, language='en'):
        if language == 'en':
            # sentence_new = self.tokenizer.tokenize(sentence)
            sentence_new = sentence.split()
        elif language == 'zh':
            sentence_new = jieba.lcut(sentence)
        return sentence_new

    def get_sentences_labels(self):
        return self.sentence_list_all, self.label_all


def gen_vocab(sentences_all, filter=False):
    vocab = {}
    for sent in sentences_all:
        for word in sent:
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    if filter:
        # 去除标点
        # 去除低频词
        pattern = re.compile(u'[\u4E00-\u9FA5|\s\w]+')
        vocab_filted = {}
        for w in vocab:
            if vocab[w] > 3 and re.match(pattern, w):
                vocab_filted[w] = vocab[w]
    else:
        vocab_filted = vocab
    return vocab_filted


def derive_keywords_scores(sentences_all, labels_all, vocab, ratio=0.3, filename='keywords.txt'):
    c = Counter(labels_all)
    print(c)
    documents = []
    for i in range(len(sentences_all)):
        documents.append((labels_all[i], set(sentences_all[i])))
    label_set = {'Uu', 'PS+', 'PS-', 'CT+', 'CT-'}
    select_num = -1
    word_scores = {}
    word_scores_type = {'Uu': {}, 'PS+': {}, 'PS-': {}, 'CT+': {}, 'CT-': {}}
    for label in label_set:
        f = selectFeatures(documents, label, top_k=select_num, vocabulary=vocab, select_type="mi")
        # print(label, f[:30])
        for w, score in f:
            if w not in word_scores:
                word_scores[w] = 0.
            word_scores[w] += score
            word_scores_type[label][w] = score
    word_scores_sorted = sorted(word_scores.items(), key=lambda k: k[1], reverse=True)
    # print('overall', word_scores_sorted[:30])
    select_words = ['tokens\toverall\tUu\tPS+\tPS-\tCT+\tCT-']
    for item in word_scores_sorted[:int(len(vocab) * ratio)]:
        select_words.append('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(item[0], item[1], word_scores_type['Uu'].get(item[0], 0.),
                                                                word_scores_type['PS+'].get(item[0], 0.),
                                                                word_scores_type['PS-'].get(item[0], 0.),
                                                                word_scores_type['CT+'].get(item[0], 0.),
                                                                word_scores_type['CT-'].get(item[0], 0.)))
    return select_words


def write(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for w in data:
            f.write(w + '\n')


def read(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


# def main():
#     # d = Data(data_path='./data/english.xml')
#     d = Data(data_path='./data/chinese.xml', language='zh')
#     sentences_all, labels_all = d.sentence_list_all, d.label_all
#     vocab = gen_vocab(sentences_all, filter=True)
#     # derive_keywords_scores(sentences_all, labels_all, vocab, ratio=0.3, filename='keywords_en.txt')
#     derive_keywords_scores(sentences_all, labels_all, vocab, ratio=0.3, filename='keywords_zh.txt')

# if __name__ == '__main__':
#     main()