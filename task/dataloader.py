import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedKFold
import torch
import torch.utils.data as D
from task.config import config
import numpy as np
from tqdm import tqdm


stop_doc = {'CD1739', 'CD1883', 'CD2187', 'CD4167', 'ED1397'}


def k_fold_split(data_path):
    train_idx, test_idx = [], []
    labels = []
    label2idx = {}

    tree = ET.parse(data_path)
    root = tree.getroot()
    for document_set in root:
        for document in document_set:
            id = document.attrib['id']
            if id not in stop_doc:
                label = document.attrib['document_level_value']
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
                labels.append(label2idx[label])
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for train, test in skf.split(np.zeros(len(labels)), labels):
        train_idx.append(train)
        test_idx.append(test)
    return train_idx, test_idx, label2idx


def collate(samples):
    id, label, trigger, trigger_mask, data, attention, \
    sent_idx, trigger_word_idx, trigger_label, sent_num,doc_position_sent, doc_mask_keywords, graph = map(list, zip(*samples))

    batched_ids = tuple(id)
    batched_labels = torch.tensor(label)
    batched_triggers = torch.cat(trigger, dim=0)
    batched_trigger_mask = torch.cat(trigger_mask, dim=0)
    batched_data = torch.cat(data, dim=0)
    batched_attention = torch.cat(attention, dim=0)
    batched_sent_idx = sent_idx
    batched_trigger_word_idx = trigger_word_idx
    batched_trigger_labels = torch.cat(trigger_label, dim=0)
    batched_sent_num = torch.tensor(sent_num)
    batched_graph = torch.cat(graph, dim=0)
    batched_doc_position_sent = torch.cat(doc_position_sent, dim=0)
    batched_doc_mask_keywords = torch.cat(doc_mask_keywords, dim=0)
    return batched_ids, batched_labels, batched_triggers, batched_trigger_mask, \
           batched_data, batched_attention, \
           batched_sent_idx, batched_trigger_word_idx, batched_trigger_labels, batched_sent_num, \
           batched_doc_position_sent, batched_doc_mask_keywords, batched_graph


def get_data(data, Data, train_idx, test_idx, label2idx, config, cv_index):
    trainset = Data(data,
                    config.data_path,
                    config.saved_path,
                    train_idx,
                    label2idx,
                    cv_index=cv_index,
                    is_training=True,
                    config=config,
                    language=config.language)
    train_loader = D.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    testset = Data(data,
                   config.data_path,
                   config.saved_path,
                   test_idx,
                   label2idx,
                   cv_index=cv_index,
                   is_training=False,
                   config=config,
                   language=config.language)
    test_loader = D.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    return train_loader, test_loader