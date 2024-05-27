import torch
from task.config import config
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
import time
import pdb
from tqdm import tqdm


class Framework:
    def __init__(self):
        pass

    def evaluate(self, model, test_loader, filepath=None):
        if filepath is not None:
            f = open(filepath, 'w')
        model.eval()
        total = 0
        correct = 0
        y_true = []
        y_pred = []
        bar_test = tqdm(test_loader)
        with torch.no_grad():
            for batch_idx, (ids, labels, triggers, trigger_masks, words, masks, sent_idx, trigger_word_idx, trigger_labels,
                            sent_nums, doc_position_sent, doc_mask_keywords, graphs) in enumerate(bar_test):
                if config.gpu:
                    triggers = triggers.cuda()
                    trigger_masks = trigger_masks.cuda()
                    words = words.cuda()
                    masks = masks.cuda()
                    # sent_idx = sent_idx.cuda()
                    # trigger_word_idx = trigger_word_idx.cuda()
                    graphs = graphs.cuda()
                    labels = labels.cuda()
                    sent_nums = sent_nums.cuda()
                    trigger_labels = trigger_labels.cuda()
                    doc_position_sent = doc_position_sent.cuda()
                    doc_mask_keywords = doc_mask_keywords.cuda()
                sent_idx_list = []
                trigger_word_idx_list = []
                for i in range(len(sent_idx)):
                    sent_idx_list.append(sent_idx[i].cuda())
                    trigger_word_idx_list.append(trigger_word_idx[i].cuda())
                predictions, trigger_predictions, main_loss, aux_loss, kl_loss = model(ids=ids,
                                                                                       triggers=triggers,
                                                                                       trigger_masks=trigger_masks,
                                                                                       words=words,
                                                                                       masks=masks,
                                                                                       sent_idx=sent_idx_list,
                                                                                       trigger_word_idx=trigger_word_idx_list,
                                                                                       sent_nums=sent_nums,
                                                                                       doc_position_sent=doc_position_sent,
                                                                                       doc_mask_keywords=doc_mask_keywords,
                                                                                       graphs=graphs)
                _, predicted = torch.max(predictions.data, 1)
                correct += predicted.data.eq(labels.data).cpu().sum()
                y_true += labels.cpu().data.numpy().tolist()
                y_pred += predicted.cpu().data.numpy().tolist()

                if filepath is not None:
                    batch = labels.shape[0]
                    for i in range(batch):
                        f.write(ids[i] + "\t" + str(labels[i].item()) + "\t" + str(predicted[i].item()) + "\n")

        f1_micro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='micro')
        f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')
        if filepath is not None:
            f.write("f1_micro: " + str(f1_micro) + "\n")
            f.write("f1_macro: " + str(f1_macro) + "\n")
            f.close()
        return f1_micro, f1_macro

    def train(self, model, trainloader, optimizer, config, epoch):
        model.train()
        # start_time = time.time()
        batch_num = len(trainloader)
        bar_train = tqdm(trainloader)
        loss_list = []
        # sentence_num, doc_position_sent, doc_mask_keywords, graph
        for batch_idx, (ids, labels, triggers, trigger_masks, words, masks, sent_idx, trigger_word_idx, trigger_labels,
                        sent_nums, doc_position_sent, doc_mask_keywords, graphs) in enumerate(bar_train):
            if config.gpu:
                triggers = triggers.cuda()
                trigger_masks = trigger_masks.cuda()
                words = words.cuda()
                masks = masks.cuda()
                # trigger_word_idx = trigger_word_idx.cuda()
                graphs = graphs.cuda()
                labels = labels.cuda()
                sent_nums = sent_nums.cuda()
                trigger_labels = trigger_labels.cuda()

                doc_position_sent = doc_position_sent.cuda()
                doc_mask_keywords = doc_mask_keywords.cuda()
            sent_idx_list = []
            trigger_word_idx_list = []
            for i in range(len(sent_idx)):
                sent_idx_list.append(sent_idx[i].cuda())
                trigger_word_idx_list.append(trigger_word_idx[i].cuda())
            optimizer.zero_grad()
            predictions, trigger_predictions, main_loss, aux_loss, kl_loss = model(ids=ids,
                                                                                   triggers=triggers,
                                                                                   trigger_masks=trigger_masks,
                                                                                   words=words,
                                                                                   masks=masks,
                                                                                   sent_idx=sent_idx_list,
                                                                                   trigger_word_idx=trigger_word_idx_list,
                                                                                   sent_nums=sent_nums,
                                                                                   graphs=graphs,
                                                                                   labels=labels,
                                                                                   trigger_labels=trigger_labels,
                                                                                   doc_position_sent=doc_position_sent,
                                                                                   doc_mask_keywords=doc_mask_keywords,
                                                                                   batch_idx=batch_idx)
            # pdb.set_trace()
            # if torch.cuda.device_count() > 1:
            #     main_loss = torch.mean(main_loss)
            #     aux_loss = torch.mean(aux_loss)
            #     kl_loss = torch.mean(kl_loss)

            loss = main_loss
            if config.kl_loss:
                loss += kl_loss
            if config.aux_loss:
                loss += aux_loss

            # print(batch_idx)
            # print(loss)
            bar_train.set_description(
                'epoch_index: {} | batch_idx: {}/{} | loss: {:.4f} | main-loss: {:.4f} | kl-loss: {:.4f} | aux-loss: {:.4f}'.
                format(epoch, batch_idx, batch_num, loss, main_loss, kl_loss, aux_loss))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            loss_list.append(loss.item())
        # print("time:%.3f" % (time.time() - start_time))
        return np.mean(loss_list)
