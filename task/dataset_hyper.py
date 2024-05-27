from ast import keyword
from collections import OrderedDict
import torch
from transformers import BertTokenizer
import xml.etree.ElementTree as ET
import numpy as np
import os
import pickle
import re
import pdb
from task.utils import read_text_data, write_text_data
from task.preprocess import derive_keywords_scores, gen_vocab, DataReader
import jieba
from tqdm import tqdm


def fetch_position(sentence,
                   keyword_score_dict,
                   max_text_length,
                   max_num=10,
                   tokenizer=None,
                   language='en',
                   cls_token="[CLS]",
                   sep_token="[SEP]",
                   mask_token="[MASK]",
                   special_tokens_count=2):
    if language == 'en':
        # 手动处理tokenization
        tokens = sentence.split()
        tokens = [token.lower() for token in tokens]
        subwords = list(map(tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        tokens = [item for indices in subwords for item in indices]
        if len(tokens) > max_text_length - special_tokens_count:
            tokens = tokens[:(max_text_length - special_tokens_count)]
        len_text = len(tokens)
        tokens = [cls_token] + tokens + [sep_token]
        token_start_idxs = np.cumsum([0] + subword_lengths) + 1  # this will lead to one more end position
        token_start_idxs = token_start_idxs.tolist()
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        at_msk = [1] * (len_text + special_tokens_count)
        seg_ids = [0] * (len_text + special_tokens_count)

        pad_len = max_text_length - len_text - special_tokens_count
        token_ids += [0] * pad_len
        at_msk += [0] * pad_len
        seg_ids += [0] * pad_len

        # 处理关键词搜索
        count = 0
        ws = sentence.split()
        position_sent = []
        # print(ws)
        for key in keyword_score_dict:
            for i, item in enumerate(ws):
                if item == key and count < max_num:  # 不能超过数量
                    subwords = tokenizer.tokenize(item)
                    n = len(subwords)
                    if token_start_idxs[i] + n - 1 < max_text_length - special_tokens_count:  # 不能超过最大句长
                        position_sent.append([token_start_idxs[i], token_start_idxs[i] + n - 1])
                        count += 1

        mask_keywords = [1] * count + [0] * (max_num - count)
        for i in range(max_num - count):
            position_sent.append([0, 0])

    elif language == 'zh':
        tokens = list(sentence)
        tokens = [token.lower() for token in tokens]
        if len(tokens) > max_text_length - special_tokens_count:
            tokens = tokens[:(max_text_length - special_tokens_count)]
        len_text = len(tokens)
        tokens = [cls_token] + tokens + [sep_token]

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        at_msk = [1] * (len_text + special_tokens_count)
        seg_ids = [0] * (len_text + special_tokens_count)

        pad_len = max_text_length - len_text - special_tokens_count
        token_ids += [0] * pad_len
        at_msk += [0] * pad_len
        seg_ids += [0] * pad_len

        words = jieba.lcut(sentence)
        word_lengths = list(map(len, words))
        token_start_idxs = np.cumsum([0] + word_lengths) + 1  # this will lead to one more end position
        # print(words)
        token_start_idxs = token_start_idxs.tolist()

        count = 0
        position_sent = []
        for key in keyword_score_dict:
            for i, item in enumerate(words):
                if item == key and count < max_num:
                    n = len(item)
                    if token_start_idxs[i] + n - 1 < max_text_length - special_tokens_count:
                        position_sent.append([token_start_idxs[i], token_start_idxs[i] + n - 1])
                        count += 1
                        # print(item, item, count)
                        # print([token_start_idxs[i], token_start_idxs[i] + n - 1])
                        # print(tokens[token_start_idxs[i]:token_start_idxs[i] + n])
                        assert ''.join(tokens[token_start_idxs[i]:token_start_idxs[i] + n]) == item

        mask_keywords = [1] * count + [0] * (max_num - count)
        for i in range(max_num - count):
            position_sent.append([0, 0])
    return token_ids, at_msk, seg_ids, position_sent, mask_keywords, token_start_idxs


class Data(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 data_path,
                 saved_data_path_root,
                 data_idx,
                 label2idx,
                 cv_index=0,
                 is_training=True,
                 config=None,
                 language='en'):
        self.is_traing = is_training
        self.data = []
        self.language = language
        self.sentence_max_length = config.max_sen_len

        self.tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
        if is_training:
            saved_data_path = saved_data_path_root + '_train_' + str(cv_index) + '.pkl'
        else:
            saved_data_path = saved_data_path_root + '_test_' + str(cv_index) + '.pkl'

        print('Reading data from {}.'.format(data_path))
        if os.path.exists(saved_data_path) and not config.force_refresh:
            with open(file=saved_data_path, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
            print('load preprocessed data from {}.'.format(saved_data_path))
        else:
            keywords_path = saved_data_path_root + '_keywords_' + str(cv_index) + '.txt'
            print('Reading keywords from {}.'.format(keywords_path))
            # pdb.set_trace()
            # 根据训练集生成关键词表
            if is_training:
                # if os.path.exists(keywords_path) and not config.force_refresh:
                if os.path.exists(keywords_path):
                    keywords = read_text_data(keywords_path)[1:]
                else:
                    print('Deriving keywords with MI')
                    dr = DataReader(data, data_idx, language)
                    sentences_all, labels_all = dr.get_sentences_labels()
                    vocab = gen_vocab(sentences_all, filter=True)
                    select_words = derive_keywords_scores(sentences_all, labels_all, vocab, ratio=config.keyword_ratio)
                    write_text_data(select_words, keywords_path)
                    keywords = select_words[1:]
            else:
                keywords = read_text_data(keywords_path)[1:]
            keyword_score_dict = OrderedDict()
            for k in keywords:
                k_spt = k.split('\t')
                keyword_score_dict[k_spt[0]] = float(k_spt[1])
            print('Done: Deriving keywords')

            data_selected = [doc for i, doc in enumerate(data) if i in data_idx]
            count = 0
            for i, doc in enumerate(tqdm(data_selected)):
                id = doc['id']  # 文档id
                label = label2idx[doc['value']]  # 文档事实标签
                sentence_list = []
                trigger_word_list = []
                doc_position_sent = []
                doc_mask_keywords = []
                for sent_id, sent in enumerate(doc['sentence']):
                    sent_text = sent['text'].lower()
                    token_ids, at_msk, seg_ids, position_sent, mask_keywords, token_start_idxs = fetch_position(
                        sent_text,
                        keyword_score_dict,
                        max_text_length=self.sentence_max_length,
                        tokenizer=self.tokenizer,
                        language=language,
                        max_num=config.max_keyword_len)
                    doc_position_sent.append(position_sent)
                    doc_mask_keywords.append(mask_keywords)
                    sent_info = {
                        'trigger_num': 0,
                        'data': torch.tensor(token_ids).unsqueeze(0),
                        'attention': torch.tensor(at_msk).unsqueeze(0)
                    }
                    if sent['trigger_num'] > 0:
                        # 如果存在 事件
                        for trigger in sent['triggers']:
                            trigger_span = trigger['span']
                            trigger_text = trigger['text'].lower()
                            if self.language == 'zh':
                                tmp = ''.join(list(sent_text)[:trigger_span[0]])
                                tmp_subwords = list(tmp)
                                trigger_subwords = list(trigger_text)
                            elif self.language == 'en':
                                tmp = ' '.join(sent_text.split()[:trigger_span[0]])
                                tmp_subwords = self.tokenizer.tokenize(tmp)  # 本句子，子词
                                trigger_subwords = self.tokenizer.tokenize(trigger_text)  # 触发词
                            else:
                                raise NotImplementedError('Unknown language')
                            pos0 = len(tmp_subwords) + 1
                            pos1 = pos0 + len(trigger_subwords)

                            if pos0 >= self.sentence_max_length - 1:
                                continue
                            if pos1 >= self.sentence_max_length - 1:
                                continue
                            if self.tokenizer.convert_ids_to_tokens(token_ids[pos0:pos1]) != trigger_subwords:
                                print(self.tokenizer.convert_ids_to_tokens(token_ids[pos0:pos1]))
                                print(trigger_subwords)
                                pdb.set_trace()
                            assert self.tokenizer.convert_ids_to_tokens(token_ids[pos0:pos1]) == trigger_subwords

                            trigger_word_idx = torch.zeros(self.sentence_max_length)
                            trigger_word_idx[pos0:pos1] = 1.0 / (pos1 - pos0)

                            trigger_word_list.append({
                                'sent_id': sent_id,
                                'idx': trigger_word_idx,
                                'value': label2idx[trigger['value']]
                            })
                            sent_info['trigger_num'] += 1
                    sentence_list.append(sent_info)
                    if len(sentence_list) >= config.max_sen_num:
                        break

                # 所有包含trigger的句子
                doc_trigger = ''
                for sent in doc['sentence']:
                    if sent['trigger_num'] > 0:
                        doc_trigger += sent['text']
                trigger_data = self.tokenizer(doc_trigger,
                                              return_tensors='pt',
                                              padding='max_length',
                                              truncation=True,
                                              max_length=260)

                # trigger_word_list_new = []
                # for tri in trigger_word_list:
                #     if tri['']

                # construct graph
                # 1 document node, 35 sentence nodes, 20 trigger nodes for english data,
                # 1 document node, 35 sentence nodes, 45 trigger nodes for chinese data
                # 句子长度：80 eng, 150 chi
                # todo: 控制节点数量和触发词的数量
                # opt.max_sen_num + opt.max_tri_num + 1
                # edge: sent-sent, men-men, sent-men, sent-doc, men-doc
                max_node_num = config.max_sen_num + config.max_tri_num + 1 + config.max_sen_num * config.max_keyword_len
                max_edge_num = config.max_tri_num + 4 + config.max_tri_num + config.max_sen_num + config.max_sen_num
                graph = self.create_rel_graph(sentence_list, trigger_word_list, doc_mask_keywords, max_node_num,
                                              max_edge_num)  # node
                # graph = [g.tolist() for g in graph]
                # edge: sent-sent(1), eve-eve(1), sent-eve(eve), doc-sent(1), doc-eve(1), key-eve(eve), key-sent(sent), key(sent)
                if config.aba == 'entire':
                    pass
                
                graph = np.array(graph)
                config.n_rel = len(graph) + 1

                if len(trigger_word_list) > 0:
                    self.data.append({
                        'ids': id,  # 文档id
                        'labels': label,  # 文档标签
                        'triggers': trigger_data['input_ids'],  # trigger 句子
                        'trigger_masks': trigger_data['attention_mask'],  # trigger 句子
                        'sentences': sentence_list,  # 句子token和mask
                        'trigger_words': trigger_word_list,  # 触发词位置及标签
                        'doc_position_sent': doc_position_sent,
                        'doc_mask_keywords': doc_mask_keywords,
                        'graphs': graph
                    })
                else:
                    print(id)
                    count += 1
                # if label not in [0, 1, 2]:
                # print(label)
            print('count: ', count)
            if language == 'en':
                # save data
                with open(file=saved_data_path, mode='wb') as fw:
                    pickle.dump({'data': self.data}, fw)
                print('finish reading {} and save preprocessed data to {}.'.format(data_path, saved_data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence_list = self.data[idx]['sentences']
        data_list = []
        attention_list = []
        for s in sentence_list:
            data_list.append(s['data'])  # token_id
            attention_list.append(s['attention'])
        data = torch.cat(data_list, dim=0)
        attention = torch.cat(attention_list, dim=0)
        sentence_num = len(data)

        trigger_word_list = self.data[idx]['trigger_words']
        sent_idx_list = []
        trigger_word_idx_list = []
        trigger_label_list = []
        for t in trigger_word_list:
            sent_idx_list.append(t['sent_id'])
            trigger_word_idx_list.append(t['idx'])
            trigger_label_list.append(t['value'])
        sent_idx = torch.tensor(sent_idx_list)
        trigger_word_idx = torch.stack(trigger_word_idx_list, dim=0)
        trigger_label = torch.tensor(trigger_label_list)
        graph = torch.tensor(self.data[idx]['graphs']).unsqueeze(0)  # 增加 bs维

        # print(self.data[idx]['doc_position_sent'])
        doc_position_sent = torch.tensor(self.data[idx]['doc_position_sent'])
        doc_mask_keywords = torch.tensor(self.data[idx]['doc_mask_keywords'])

        return self.data[idx]['ids'], \
               torch.tensor(self.data[idx]['labels'], dtype=torch.long), \
               self.data[idx]['triggers'], \
               self.data[idx]['trigger_masks'], \
               data, \
               attention, \
               sent_idx, \
               trigger_word_idx, \
               trigger_label, \
               sentence_num, doc_position_sent, doc_mask_keywords, graph

    def create_rel_graph(self, sentence_list, trigger_word_list, doc_mask_keywords, n_node, n_edge):
        # node: doc, sent, eve, key
        # edge: sent-sent(1), eve-eve(1), sent-eve(eve), doc-sent(1), doc-eve(1), key-eve(eve), key-sent(sent), key(sent)
        graph_1 = np.zeros((n_edge, n_node), dtype=np.int8)  # 句子
        graph_2 = np.zeros((n_edge, n_node), dtype=np.int8)  # 事件
        graph_3 = np.zeros((n_edge, n_node), dtype=np.int8)  # 句子-事件
        graph_4 = np.zeros((n_edge, n_node), dtype=np.int8)  # 文档-句子
        graph_5 = np.zeros((n_edge, n_node), dtype=np.int8)  # 文档-事件

        graph_6 = np.zeros((n_edge, n_node), dtype=np.int8)  # 关键词-事件
        graph_7 = np.zeros((n_edge, n_node), dtype=np.int8)  # 关键词-句子
        graph_8 = np.zeros((n_edge, n_node), dtype=np.int8)  # 关键词

        sent_num = len(sentence_list)  # 所有句子
        trigger_num = len(trigger_word_list)  # 包含触发词的句子
        key_num = len(doc_mask_keywords[0])
        # pdb.set_trace()
        for i in range(sent_num):
            graph_1[0][1 + i] = 1
            graph_4[2 + trigger_num][1 + i] = 1  # 句子
            graph_4[2 + trigger_num][0] = 1  # 文档
            graph_7[2 + trigger_num + 2 + trigger_num + i][1 + i] = 1

        for i in range(trigger_num):
            graph_2[1][1 + sent_num + i] = 1
            graph_3[2 + i][1 + trigger_word_list[i]['sent_id']] = 1  # 句子
            graph_3[2 + i][1 + sent_num + i] = 1  # 触发词
            graph_5[2 + trigger_num + 1][1 + sent_num + i] = 1  # 触发词
            graph_5[2 + trigger_num + 1][0] = 1  # 文档
            graph_6[2 + trigger_num + 2 + i][1 + sent_num + i] = 1  # 关键词-事件

        for i in range(sent_num):
            for j in range(trigger_num):
                for k in range(key_num):
                    graph_6[2 + trigger_num + 2 + j][1 + sent_num + trigger_num + key_num * trigger_word_list[j]['sent_id'] +
                                                     k] = doc_mask_keywords[trigger_word_list[j]['sent_id']][k]
                    graph_7[2 + trigger_num + 2 + trigger_num + i][1 + sent_num + trigger_num + key_num * i +
                                                                   k] = doc_mask_keywords[i][k]
                    # print(sent_num, trigger_num, key_num, len(doc_mask_keywords),
                    #       2 + trigger_num + sent_num + 2 + trigger_num + sent_num + i,
                    #       1 + sent_num + trigger_num + key_num * i + k, n_edge, n_node)
                    graph_8[2 + trigger_num + 2 + trigger_num + sent_num + i][1 + sent_num + trigger_num + key_num * i +
                                                                              k] = doc_mask_keywords[i][k]

        graph = np.array([graph_1, graph_2, graph_3, graph_4, graph_5, graph_6, graph_7, graph_8])
        return graph
