from distutils.command.config import config
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch
import torch.nn.functional as F
import pdb

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None

    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)


class CGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(CGCN, self).__init__()
        self.gcn_layer1 = GCNLayer(in_dim, hid_dim)
        self.gcn_layer2 = GCNLayer(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.trans_mean = nn.Linear(in_dim, in_dim)

    def forward(self, adj, gcn_inputs):
        output_features = []
        output_features.append(gcn_inputs)
        adj_list = []
        for i in range(adj.size()[0]):
            adj_list.append(self.normalize(adj[i].view(adj.size()[1], adj.size()[2])))
        adj = torch.cat(adj_list, dim=0)
        adj = adj.type_as(gcn_inputs)
        mean_vectors = F.relu(self.trans_mean(gcn_inputs))
        output_features.append(mean_vectors)

        x_mean = mean_vectors
        Ax_mean = adj.bmm(x_mean)
        hid_output_mean = self.gcn_layer1(Ax_mean)
        output_features.append(hid_output_mean)

        Ax_mean = adj.bmm(hid_output_mean)
        output_mean = self.gcn_layer2(Ax_mean)
        # sample_v = torch.randn(1, 1)[0][0]
        # output_mean = output_mean + (torch.sqrt(output_var + 1e-8) * sample_v)
        output_features.append(output_mean)
        out_vars = []
        return output_features, out_vars

    def normalize(self, A, symmetric=True):
        # A = A+I
        A = A + torch.eye(A.size(0)).cuda()
        # degree of nodes
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D).unsqueeze(0)

    def normalize1(self, A, symmetric=True):
        # A = A+I
        A = A + torch.eye(A.size(0)).cuda()
        # degree of nodes
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A).mm(D).unsqueeze(0)


class RGCN_v1(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(RGCN_v1, self).__init__()
        self.gcn_layer1 = GCNLayer(in_dim * 4, hid_dim)
        self.gcn_layer2 = GCNLayer(hid_dim * 4, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.trans_mean = nn.Linear(in_dim, in_dim)

    def forward(self, adj, gcn_inputs):
        output_features = []
        output_features.append(gcn_inputs)
        adj = adj.type_as(gcn_inputs)  # [b, r, n, n]
        mean_vectors = F.relu(self.trans_mean(gcn_inputs))  # [b, n, e]
        output_features.append(mean_vectors)

        b, r, n = adj.size(0), adj.size(1), adj.size(2)
        x_mean = mean_vectors
        Ax_mean = torch.einsum('brij,bjk->birk', adj, x_mean)  # [b,n,r,e]
        Ax_mean = Ax_mean.view(b, n, -1)
        hid_output_mean = self.gcn_layer1(Ax_mean)
        output_features.append(hid_output_mean)

        # pdb.set_trace()
        Ax_mean = torch.einsum('brij,bjk->birk', adj, hid_output_mean)  # [b,n,r,e]
        Ax_mean = Ax_mean.view(b, n, -1)
        output_mean = self.gcn_layer2(Ax_mean)
        output_features.append(output_mean)
        out_vars = []
        return output_features, out_vars


class HGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout, n_rel=5):
        super(HGAT, self).__init__()
        self.gcn_layer1_edge = GCNLayer(in_dim * n_rel, hid_dim)
        self.gcn_layer1_node = GCNLayer(in_dim * n_rel, hid_dim)

        self.gcn_layer2_edge = GCNLayer(in_dim * n_rel, hid_dim)
        self.gcn_layer2_node = GCNLayer(in_dim * n_rel, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.trans_mean = nn.Linear(in_dim, in_dim)

    def forward(self, adj, gcn_inputs):
        '''
        gcn_inputs: [b, nn, e]
        adj: [b, r, ne, nn], 边数*节点数
        无向图
        '''

        output_features = []
        output_features.append(gcn_inputs)
        adj = adj.type_as(gcn_inputs)  # [b, r, ed, nd], ed for edge, nd for node

        mean_vectors = F.relu(self.trans_mean(gcn_inputs))  # [b, n, e]
        output_features.append(mean_vectors)

        b, r, ne, nn = adj.size(0), adj.size(1), adj.size(2), adj.size(3)

        # 第一次
        x_mean = mean_vectors
        Ax_mean = torch.einsum('brij,bjk->birk', adj, x_mean)  # [b,ed,r,e]
        Ax_mean = Ax_mean.view(b, ne, -1)
        hid_edge_mean = self.gcn_layer1_edge(Ax_mean)  # [b, ed, r*e]

        Ax_node_mean = torch.einsum('brij,bik->bjrk', adj, hid_edge_mean)
        Ax_node_mean = Ax_node_mean.view(b, nn, -1)
        hid_node_mean = self.gcn_layer1_node(Ax_node_mean)  # [b, ed, r*e]
        output_features.append(hid_node_mean)

        # if torch.isnan(hid_edge_mean).sum():
        #     pdb.set_trace()

        # 第二次
        x_mean = hid_node_mean
        Ax_mean = torch.einsum('brij,bjk->birk', adj, x_mean)  # [b,ed,r,e]
        Ax_mean = Ax_mean.view(b, ne, -1)
        hid_edge_mean = self.gcn_layer2_edge(Ax_mean)  # [b, ed, r*e]

        Ax_node_mean = torch.einsum('brij,bik->bjrk', adj, hid_edge_mean)
        Ax_node_mean = Ax_node_mean.view(b, nn, -1)
        hid_node_mean = self.gcn_layer2_node(Ax_node_mean)  # [b, ed, r*e]
        output_features.append(hid_node_mean)

        out_vars = []
        return output_features, out_vars


class UHGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout, n_rel=5, norm=False, config=None):
        super(UHGAT, self).__init__()
        self.gcn_layer1_edge = GCNLayer(in_dim * (n_rel - 1), hid_dim)
        self.gcn_layer1_node = GCNLayer(in_dim * n_rel, hid_dim)
        self.gcn_layer1_edge_var = GCNLayer(in_dim * (n_rel - 1), hid_dim)
        self.gcn_layer1_node_var = GCNLayer(in_dim * n_rel, hid_dim)

        self.gcn_layer2_edge = GCNLayer(in_dim * (n_rel - 1), hid_dim)
        self.gcn_layer2_node = GCNLayer(in_dim * n_rel, hid_dim)
        self.gcn_layer2_edge_var = GCNLayer(in_dim * (n_rel - 1), hid_dim)
        self.gcn_layer2_node_var = GCNLayer(in_dim * n_rel, hid_dim)

        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.trans_mean = nn.Linear(in_dim, in_dim)
        self.trans_var = nn.Linear(in_dim, in_dim)
        self.config = config

    def forward(self, adj, gcn_inputs):
        '''
        gcn_inputs: [b, nn, e]
        adj: [b, r, ne, nn], 边数*节点数
        假设无向图
        '''

        adj = adj.type_as(gcn_inputs)  # [b, r, ed, nd], ed for edge, nd for node

        output_features = []
        output_features.append(gcn_inputs)
        mean_vectors = F.relu(self.trans_mean(gcn_inputs))  # [b, n, e]
        output_features.append(mean_vectors)

        out_vars = []
        out_vars.append(gcn_inputs)
        var_vectors = F.relu(self.trans_var(gcn_inputs))
        out_vars.append(var_vectors)

        # 第一次
        hid_node_mean, hid_node_var = self.uncertain_hypergraph_conv(adj, mean_vectors, var_vectors, self.gcn_layer1_edge,
                                                                     self.gcn_layer1_edge_var, self.gcn_layer1_node,
                                                                     self.gcn_layer1_node_var)
        output_features.append(hid_node_mean)
        out_vars.append(hid_node_var)
        # 第二次
        hid_node_mean, hid_node_var = self.uncertain_hypergraph_conv(adj, hid_node_mean, hid_node_var, self.gcn_layer2_edge,
                                                                     self.gcn_layer2_edge_var, self.gcn_layer2_node,
                                                                     self.gcn_layer2_node_var)
        # 利用期望来进行推断
        output_mean = hid_node_mean
        output_var = hid_node_var
        if self.training:
            sample_v = torch.randn(1, 1)[0][0]
            output_mean = output_mean + (torch.sqrt(output_var + 1e-8) * sample_v)
        output_features.append(output_mean)  #重参数化
        out_vars.append(output_var)
        return output_features, out_vars

    def uncertain_hypergraph_conv(self, adj_ori, mean_vectors, var_vectors, gcn_layer_edge, gcn_layer_edge_var, gcn_layer_node,
                                  gcn_layer_node_var):
        b, r, ne, nn = adj_ori.size(0), adj_ori.size(1), adj_ori.size(2), adj_ori.size(3)

        # 节点到边
        if self.norm:
            adj = adj_ori / (adj_ori.sum(dim=-1, keepdim=True) + 1e-8)
            adj_var = adj * adj
        else:
            adj = adj_ori
            adj_var = adj * adj

        node_weight = torch.exp(-self.config.alpha * var_vectors)
        x_mean = mean_vectors.mul(node_weight)
        x_var = var_vectors.mul(node_weight).mul(node_weight)

        Ax_mean = torch.einsum('brij,bjk->birk', adj, x_mean)  # [b,ed,r,e]
        Ax_mean = Ax_mean.view(b, ne, -1)
        hid_edge_mean = gcn_layer_edge(Ax_mean)  # [b, ed, r*e]

        Ax_var = torch.einsum('brij,bjk->birk', adj_var, x_var)  # [b,n,r,e]
        Ax_var = Ax_var.view(b, ne, -1)
        hid_edge_var = gcn_layer_edge_var(Ax_var)  # [b, ed, r*e]

        # 边到节点
        if self.norm:
            adj = adj_ori / (adj_ori.sum(dim=-2, keepdim=True) + 1e-8)
            adj_var = adj * adj
        else:
            adj = adj_ori
            adj_var = adj * adj
        node_weight = torch.exp(-self.config.alpha * hid_edge_var)
        hid_edge_mean = hid_edge_mean.mul(node_weight)
        hid_edge_var = hid_edge_var.mul(node_weight).mul(node_weight)

        Ax_node_mean = torch.einsum('brij,bik->bjrk', adj, hid_edge_mean)
        Ax_node_mean = Ax_node_mean.view(b, nn, -1)
        Ax_node_mean = torch.cat([Ax_node_mean, x_mean], dim=-1)  # 自身的边
        hid_node_mean = gcn_layer_node(Ax_node_mean)  # [b, ed, r*e]

        Ax_node_var = torch.einsum('brij,bik->bjrk', adj_var, hid_edge_var)
        Ax_node_var = Ax_node_var.view(b, nn, -1)
        Ax_node_var = torch.cat([Ax_node_var, x_var], dim=-1)  # 自身的边
        hid_node_var = gcn_layer_node_var(Ax_node_var)  # [b, ed, r*e]
        return hid_node_mean, hid_node_var


class GCN_Joint_EFP(nn.Module):
    def __init__(self, config, y_num):
        super(GCN_Joint_EFP, self).__init__()
        self.config = config
        self.y_num = y_num
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.bert = BertModel.from_pretrained(
            config.model_name_or_path)  # bert-base-uncased for english data, bert-base-chinese for chinese data
        print('The number of parameters of bert: ', sum(p.numel() for p in self.bert.parameters() if p.requires_grad))

        self.gcn_in_dim = config.bert_hid_size
        self.gcn_hid_dim = config.gcn_hid_dim
        self.gcn_out_dim = config.gcn_out_dim
        self.dropout = config.dropout

        self.gnn = UHGAT(self.gcn_in_dim,
                         self.gcn_hid_dim,
                         self.gcn_out_dim,
                         self.dropout,
                         n_rel=config.n_rel,
                         norm=config.norm_adj,
                         config=config)

        self.bank_size = self.gcn_in_dim + self.gcn_hid_dim + self.gcn_out_dim
        self.linear_dim = config.linear_dim
        self.predict = nn.Linear(self.bank_size, self.y_num)
        self.trigger_predict = nn.Linear(self.bank_size, self.y_num)

    def forward(self, **params):
        # print('forwarding...')

        triggers = params['triggers']  # [bs, ]
        trigger_masks = params['trigger_masks']
        bsz = triggers.size()[0]
        doc_outputs = self.bert(triggers, attention_mask=trigger_masks)  # 0为token，1为cls
        document_cls = doc_outputs[1]

        words = params['words']  # [bsz, seq_len]
        masks = params['masks']  # [bsz, seq_len]
        sent_outputs = self.bert(words, attention_mask=masks)  # sentence_cls: [bsz, bert_dim]
        sentence_embed = sent_outputs[0]  # [b, t, e]
        sentence_cls = sent_outputs[1]
        # print(document_cls)
        # pdb.set_trace()

        sent_idx = params['sent_idx']  # bsz * [trigger_num]
        trigger_word_idx = params['trigger_word_idx']  # bsz * [trigger_num, seq_len], 保存触发词位置
        graphs = params['graphs']
        assert graphs.size()[0] == bsz, "batch size inconsistent"

        split_sizes = params['sent_nums'].tolist()  # 句子数目

        feature_list = list(torch.split(sentence_cls, split_sizes, dim=0))  # bsz * [n, e]
        sentence_embed_list = list(torch.split(sentence_embed, split_sizes, dim=0))  # bsz * [n, t, e]

        doc_position_sent = params['doc_position_sent']  # [bns, 10, 2]
        doc_mask_keywords = params['doc_mask_keywords']  # [bns, 10, 2]

        doc_position_sent_list = list(torch.split(doc_position_sent, split_sizes, dim=0))
        doc_mask_keywords_list = list(torch.split(doc_mask_keywords, split_sizes, dim=0))

        # print(doc_mask_keywords)
        sentence_trigger = []
        trigger_nums = []
        for i in range(bsz):
            # 找出有触发词的句子
            t = sentence_embed_list[i].index_select(0, sent_idx[i])  # [trigger_num, seq_len, bert_dim]
            # 从句子中，得到触发词的表示
            trigger_embed = torch.sum(trigger_word_idx[i].unsqueeze(-1) * t, dim=1)  # [trigger_num, bert_dim]
            trigger_nums.append(trigger_embed.size()[0])
            # 句表示、触发词表示、pad
            fea = torch.cat((feature_list[i], trigger_embed), dim=0)

            doc_mask_keywords = doc_mask_keywords_list[i]  # [ns, 10]
            doc_position_sent = doc_position_sent_list[i]
            doc_sentence_emb = sentence_embed_list[i]
            span_doc_list = []

            for j, sent in enumerate(doc_position_sent):
                span_sent_list = []
                for k, span in enumerate(sent):
                    span_emb = doc_sentence_emb[j][span[0]:span[1] + 1]  # [ns, , e]; 
                    span_emb = torch.mean(span_emb, dim=0)  # [e]
                    span_sent_list.append(span_emb.unsqueeze(0))

                span_sent_emb = torch.cat(span_sent_list, dim=0)  # [10, e]
                span_doc_list.append(span_sent_emb.unsqueeze(0))

            span_doc_emb = torch.cat(span_doc_list, dim=0)  #[ns, 10, e]
            fea_keywords = span_doc_emb * doc_mask_keywords.unsqueeze(-1)  # [ns, 10, e] * [ns, 10, 1]
            fea_keywords_flat = fea_keywords.view(-1, fea_keywords.size(-1))
            fea = torch.cat((fea, fea_keywords_flat), dim=0)

            pad = torch.zeros(graphs.size()[-1] - 1 - fea.size()[0], fea.size()[-1]).cuda()
            fea = torch.cat((fea, pad), dim=0)
            assert fea.size()[0] == graphs.size()[-1] - 1
            sentence_trigger.append(fea.unsqueeze(0))

        sentence_trigger = torch.cat(sentence_trigger, dim=0)
        features = torch.cat((document_cls.unsqueeze(1), sentence_trigger), dim=1)
        assert features.size()[0] == bsz
        assert features.size()[1] == graphs.size()[-1]

        output_features, output_vars = self.gnn(graphs, features)  # [bsz, num_node, dim]

        output_feature_list = [output_features[0], output_features[2], output_features[3]]  # bert，第一次图，最后一次重参数化
        output_feature = torch.cat(output_feature_list, dim=-1)
        document_features = []
        trigger_features = []
        for i in range(bsz):
            document_features.append(output_feature[i:i + 1, 0, :])
            trigger_start = 1 + split_sizes[i]
            trigger_end = trigger_start + trigger_nums[i]
            trigger_features.append(output_feature[i:i + 1, trigger_start:trigger_end, :].view(-1, output_feature.size()[-1]))

        document_feature = torch.cat(document_features, dim=0).view(-1, output_feature.size()[-1])
        trigger_feature = torch.cat(trigger_features, dim=0).view(-1, output_feature.size()[-1])

        # classification
        predictions = self.predict(document_feature)
        trigger_predictions = self.trigger_predict(trigger_feature)

        main_loss, aux_loss, KL_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        if self.training:
            labels = params['labels']
            trigger_labels = params['trigger_labels']
            main_loss = nn.functional.cross_entropy(predictions, labels)
            aux_loss = nn.functional.cross_entropy(trigger_predictions, trigger_labels)

            mean = output_features[1]  # (bsz, node_num, gcn_out_dim)
            var = output_vars[1]
            KL_divergence = 0.5 * torch.mean(torch.square(mean) + var - torch.log(1e-8 + var) - 1, dim=-1)
            KL_divergence = torch.mean(KL_divergence)
            KL_loss = 5e-4 * KL_divergence

        return predictions, trigger_predictions, main_loss, aux_loss, KL_loss
