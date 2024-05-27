import imp
from task.config import config
from transformers import AdamW
import os
import numpy as np
from sklearn.metrics import f1_score
import time
from task.utils import seed_everything
import pdb
from tqdm import tqdm
from task.utils import save_checkpoint, load_checkpoint, write_jsonl, read_jsonl
from task.preprocess import Preprocess
import io  
import sys  
import torch
 


CV_NUM = 10

def overall_metric(output_path, f1_micro_list, f1_macro_list):
    # 计算十折之后的模型性能
    output = open(output_path + "/overall.txt", "w")
    f1_micro_a = np.mean(f1_micro_list)
    f1_macro_a = np.mean(f1_macro_list)
    output.write("batch_size=" + str(config.batch_size) + "\n")
    output.write("lr=" + str(config.lr) + "\n")
    output.write("f1_micro_a: " + str(f1_micro_a) + "\n")
    output.write("f1_macro_a: " + str(f1_macro_a) + "\n")
    print("F1_micro_a: %.2f F1_macro_a: %.2f" % (f1_micro_a * 100, f1_macro_a * 100))

    ct_p = []
    ct_m = []
    ps_p = []
    for i in range(CV_NUM):
        filename = output_path + "/output_" + str(i) + ".txt"
        y_true = []
        y_pred = []
        with open(filename, "r") as f:
            for l in f.readlines():
                line = l.split()
                if len(line) < 3:
                    break
                y_true.append(line[1])  # 1为真实，2为预测
                y_pred.append(line[2])

        with open(filename, "a") as f:
            t_ct_p = f1_score(y_true, y_pred, labels=[0], average="macro")
            f.write("CT+: " + str(t_ct_p) + "\n")
            ct_p.append(t_ct_p)

            t_ct_m = f1_score(y_true, y_pred, labels=[1], average="macro")
            f.write("CT-: " + str(t_ct_m) + "\n")
            ct_m.append(t_ct_m)

            t_ps_p = f1_score(y_true, y_pred, labels=[2], average="macro")
            f.write("PS+: " + str(t_ps_p) + "\n")
            ps_p.append(t_ps_p)

    output.write("CT+: " + str(np.mean(ct_p)) + "\n")
    output.write("CT-: " + str(np.mean(ct_m)) + "\n")
    output.write("PS+: " + str(np.mean(ps_p)) + "\n")
    output.close()

    print('CT+:{:.4}, CT-:{:.4}, PS+:{:.4}, f1_micro:{:.4}, f1_macro:{:.4}'.format(np.mean(ct_p), np.mean(ct_m), np.mean(ps_p),
                                                                                   f1_micro_a, f1_macro_a))


def exp_select_variant():
    from task.dataset_hyper import Data
    DATA = Data
    from models.model import GCN_Joint_EFP
    MODEL = GCN_Joint_EFP
    return MODEL, DATA


def main():
    MODEL, DATA = exp_select_variant()

    from task.dataloader import get_data, k_fold_split
    from task.framework import Framework

    if not os.path.exists(config.saved_path):
        os.mkdir(config.saved_path)

    config.path = config.path + '/' + config.variant + '_' + config.exp_name + '_' + config.aba
    config.saved_path = config.saved_path + '/' + config.variant + '_' + config.exp_name + config.aba

    config.max_n_node = config.max_sen_num + config.max_tri_num + 1
    if not os.path.exists(config.path):
        os.mkdir(config.path)

    if not os.path.exists(config.saved_path):
        os.mkdir(config.saved_path)

    seed_everything(config.seed)
    framework = Framework()

    if not os.path.exists('./data/processed'):
        os.mkdir('./data/processed')

    if not os.path.exists('./data/processed/' + config.data_path.split('/')[-1].split('.')[0] + '.jsonl'):
        pre = Preprocess()
        data = pre.preprocess(config.data_path, language=config.language)
        write_jsonl('./data/processed/' + config.data_path.split('/')[-1].split('.')[0] + '.jsonl', data)
    data = read_jsonl('./data/processed/' + config.data_path.split('/')[-1].split('.')[0] + '.jsonl')

    # 以下进行10折交叉检验，然后计算评价指标
    print('CV...')
    train_idx, test_idx, label2idx = k_fold_split(config.data_path)

    f1_micro_list = []
    f1_macro_list = []
    for i in range(CV_NUM):
        print('CV: {}'.format(i))
        fname_model = config.path + "/model_" + str(i) + ".pt"
        fname_output = config.path + "/output_" + str(i) + ".txt"

        train_loader, test_loader = get_data(data, DATA, train_idx[i], test_idx[i], label2idx, config, i)
        model = MODEL(config, len(label2idx))

        if config.gpu:
            model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=config.lr)

        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)

        max_f1 = 0
        max_f1_micro = 0
        max_f1_macro = 0
        if config.training:
            print('training begin...')
            for epoch in range(config.n_epochs):
                train_loss = framework.train(model, train_loader, optimizer, config, epoch)
                test_f1_micro, test_f1_macro = framework.evaluate(model, test_loader)
                print("Epoch:%d-%d loss:%f F1_micro:%.2f F1_macro:%.2f" %
                      (i, epoch, train_loss, test_f1_micro * 100, test_f1_macro * 100))
                # 保存最好
                if test_f1_micro + test_f1_macro > max_f1:
                    max_f1 = test_f1_micro + test_f1_macro
                    save_checkpoint(fname_model, epoch, model, optimizer)

        # 最终验证
        checkpoint, model, optimizer = load_checkpoint(fname_model, model, optimizer)
        test_f1_micro, test_f1_macro = framework.evaluate(model, test_loader, filepath=fname_output)
        print("Epoch:%d-%d F1_micro:%.6f F1_macro:%.6f" % (i, checkpoint['epoch'], test_f1_micro * 100, test_f1_macro * 100))
        f1_micro_list.append(test_f1_micro)
        f1_macro_list.append(test_f1_macro)

    overall_metric(config.path, f1_micro_list, f1_macro_list)


if __name__ == '__main__':
    main()