import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class Embedding_init(nn.Module):
    @staticmethod
    def init_emb(row, col):
        w = torch.empty(row, col)
        torch.nn.init.normal_(w)
        w = torch.nn.functional.normalize(w)
        entities_emb = nn.Parameter(w)
        return entities_emb

class GatedFusion(nn.Module):
    def __init__(self, input_size):
        super(GatedFusion, self).__init__()
        self.gate = nn.Linear(input_size * 2, input_size)
    def forward(self, feature_a, feature_b):
        # 将两个特征向量连接起来
        combined_features = torch.cat((feature_a, feature_b), dim=1)
        # 使用sigmoid函数计算门控信号
        gate_signal = torch.sigmoid(self.gate(combined_features))
        # print("gate_signal: ", gate_signal)
        # 使用门控信号来融合两个特征向量
        fused_feature = gate_signal * feature_a + (1 - gate_signal) * feature_b
        return fused_feature

class OverAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size, rel_hidden,
                 time_size,time_sizeT,
                 triple_size,
                 rel_matrix,
                 ent_matrix,
                 time_matrix,time_matrixT,
                 dropout_rate=0, depth=2, dropout_time=0.5,
                 device='cpu'
                 ):
        super(OverAll, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_time = dropout_time

        # new adding
        # rel_or_time in GraphAttention.forward

        self.e_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                        dim=node_hidden)
        self.r_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                        dim=node_hidden)

        self.t_encoder = GraphAttention2(node_size, rel_size, triple_size, time_size, time_sizeT, depth=depth, device=device,
                                        dim=node_hidden)

        self.ent_adj = self.get_spares_matrix_by_index(ent_matrix, (node_size, node_size))
        self.rel_adj = self.get_spares_matrix_by_index(rel_matrix, (node_size, rel_size))
        self.time_adj = self.get_spares_matrix_by_index(time_matrix, (node_size, time_size))
        self.time_adjT = self.get_spares_matrix_by_index(time_matrixT, (node_size, time_sizeT))


        self.ent_emb = self.init_emb(node_size, node_hidden)
        self.rel_emb = self.init_emb(rel_size, node_hidden)
        self.time_emb = self.init_emb(time_size, node_hidden)
        self.time_embT = self.init_emb(time_sizeT, node_hidden)


        self.try_emb = self.init_emb(1, node_hidden)
        self.device = device
        self.ent_adj, self.rel_adj, self.time_adj, self.time_adjT = \
            map(lambda x: x.to(device), [self.ent_adj, self.rel_adj, self.time_adj, self.time_adjT])

        # 添加门控融合模块
        self.gated_fusion = GatedFusion(node_hidden * (depth + 1))
        # 添加门控融合模块
        self.gated_fusion_time = GatedFusion(node_hidden * (depth + 1))

    # get prepared
    @staticmethod
    def get_spares_matrix_by_index(index, size):
        index = torch.LongTensor(index)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1),
                                       torch.ones_like(index[:, 0], dtype=torch.float), size)
        # dim ??
        return torch.sparse.softmax(adj, dim=1)

    @staticmethod
    def init_emb(*size):
        entities_emb = nn.Parameter(torch.randn(size))
        torch.nn.init.xavier_normal_(entities_emb)
        return entities_emb

    def forward(self, inputs):
        # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_pairs]
        ent_feature = torch.matmul(self.ent_adj, self.ent_emb)
        rel_feature = torch.matmul(self.rel_adj, self.rel_emb)
        time_feature = torch.matmul(self.time_adj, self.time_emb)
        time_featureT = torch.matmul(self.time_adjT, self.time_embT)

        # ent_feature = torch.cat([ent_feature, rel_feature, time_feature], dim=1)

        # note that time_feature and rel_feature is has the same shape of ent_feature
        # the dim = node_hidden, the shape[0] = # of entities
        # They are obtained by gather the linked rel/time of an entity

        adj_input = inputs[0]
        r_index = inputs[1]
        r_val = inputs[2]
        t_index = inputs[3]
        t_indexT = inputs[6]

        opt = [self.rel_emb, adj_input, r_index, r_val]
        opt2 = [self.time_emb, adj_input, t_index, r_val]
        opt3 = [self.time_embT, adj_input, t_indexT, r_val]

        # attention opt_1 or 2
        out_feature_ent = self.e_encoder([ent_feature] + opt)
        out_feature_rel = self.r_encoder([rel_feature] + opt)
        # out_feature_time = self.e_encoder([time_feature] + opt2, 1)
        # out_feature_timeT = self.t_encoder([time_featureT] + opt3, 1)
        out_feature_timeT = self.t_encoder([time_feature] + opt2 + [time_featureT] + opt3, 1)


        # torch.save(out_feature_ent, 'ent_emb.npy')
        # torch.save(out_feature_rel, 'rel_emb.npy')
        # torch.save(out_feature_time, 'time_emb.npy')



        # out_feature_ent2 = self.e_encoder([ent_feature] + opt)
        # out_feature_rel2 = self.r_encoder([rel_feature] + opt)
        # out_feature_time2 = self.t_encoder([time_feature] + opt2, 1)
        # out_feature_time = self.e_encoder([time_feature] + opt)

        # out_feature_time = F.dropout(out_feature_time, p=self.dropout_time, training=self.training)

        # out_feature_time = self.gated_fusion(out_feature_time, out_feature_timeT)

        # out_feature_overall = (out_feature_rel + out_feature_time) / 2

        # out_feature_overall = self.gated_fusion(out_feature_rel, out_feature_timeT)
        out_feature_overall = torch.cat((out_feature_rel, out_feature_timeT), dim=-1)

        out_feature = torch.cat((out_feature_ent, out_feature_overall), dim=-1)

        out_feature = F.dropout(out_feature, p=self.dropout_rate, training=self.training)
        return out_feature


class GraphAttention(nn.Module):
    def __init__(self, node_size, rel_size, triple_size, time_size,
                 activation=torch.tanh, use_bias=True,
                 attn_heads=1, dim=100,
                 depth=1, device='cpu'):
        super(GraphAttention, self).__init__()
        self.node_size = node_size
        self.activation = activation
        self.rel_size = rel_size

        self.time_size = time_size
        # self.time_sizeT = time_sizeT

        self.triple_size = triple_size
        self.use_bias = use_bias
        self.attn_heads = attn_heads
        self.attn_heads_reduction = 'concat'
        self.depth = depth
        self.device = device
        self.attn_kernels = []

        node_F = dim
        rel_F = dim
        self.ent_F = node_F
        ent_F = self.ent_F

        # gate kernel Eq 9 M
        self.gate_kernel = OverAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))
        self.proxy = OverAll.init_emb(64, node_F * (self.depth + 1))
        if self.use_bias:
            self.bias = OverAll.init_emb(1, ent_F * (self.depth + 1))
        for d in range(self.depth):
            self.attn_kernels.append([])
            for h in range(self.attn_heads):
                attn_kernel = OverAll.init_emb(node_F, 1)
                self.attn_kernels[d].append(attn_kernel.to(device))

    def forward(self, inputs, rel_or_time=0):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj_index = inputs[2]  # adj
        index = torch.tensor(adj_index, dtype=torch.int64)
        index = index.to(self.device)
        # adj = torch.sparse.FloatTensor(torch.LongTensor(index),
        #                                torch.FloatTensor(torch.ones_like(index[:,0])),
        #                                (self.node_size, self.node_size))
        sparse_indices = inputs[3]  # relation index  i.e. r_index
        sparse_val = inputs[4]  # relation value  i.e. r_val

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]
                ####  rel or time, 0 is rel ,else is time
                col = self.rel_size if rel_or_time == 0 else self.time_size

                rels_sum = torch.sparse.FloatTensor(
                    torch.transpose(torch.LongTensor(sparse_indices), 0, 1),
                    torch.FloatTensor(sparse_val),
                    (self.triple_size, col)
                )  # relation matrix
                rels_sum = rels_sum.to(self.device)
                rels_sum = torch.matmul(rels_sum, rel_emb)
                neighs = features[index[:, 1]]
                # selfs = features[index[:, 0]]
                rels_sum = F.normalize(rels_sum, p=2, dim=1)
                neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum

                # Eq.3
                att1 = torch.squeeze(torch.matmul(rels_sum, attention_kernel), dim=-1)
                att = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), att1, (self.node_size, self.node_size))
                # ??? dim ??
                att = torch.sparse.softmax(att, dim=1)
                # ?
                # print(att1)
                # print(att.data)
                new_features = torch_scatter.scatter_add(
                    torch.transpose(neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), 0, 1),
                    index[:, 0])
                new_features = torch.transpose(new_features, 0, 1)
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list)

            features = self.activation(features)
            outputs.append(features)


        outputs = torch.cat(outputs, dim=1)
        # proxy_att = torch.matmul(F.normalize(outputs, dim=-1),
        #                          torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
        # proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
        # proxy_feature = outputs - torch.matmul(proxy_att, self.proxy)
        #
        # if self.use_bias:
        #     gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel) + self.bias)
        # else:
        #     gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))
        # outputs = gate_rate * outputs + (1 - gate_rate) * proxy_feature
        return outputs


class GraphAttention2(nn.Module):
    def __init__(self, node_size, rel_size, triple_size, time_size,time_sizeT,
                 activation=torch.tanh, use_bias=True,
                 attn_heads=1, dim=100,
                 depth=1, device='cpu'):
        super(GraphAttention2, self).__init__()
        self.node_size = node_size
        self.activation = activation
        self.rel_size = rel_size

        self.time_size = time_size
        self.time_sizeT = time_sizeT

        self.triple_size = triple_size
        self.use_bias = use_bias
        self.attn_heads = attn_heads
        self.attn_heads_reduction = 'concat'
        self.depth = depth
        self.device = device
        self.attn_kernels = []

        node_F = dim
        rel_F = dim
        self.ent_F = node_F
        ent_F = self.ent_F

        # gate kernel Eq 9 M
        self.gate_kernel = OverAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))
        self.gate_kernel2 = OverAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))
        self.gate_kernel3 = OverAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))

        self.proxy = OverAll.init_emb(64, node_F * (self.depth + 1))
        if self.use_bias:
            self.bias = OverAll.init_emb(1, ent_F * (self.depth + 1))
        for d in range(self.depth):
            self.attn_kernels.append([])
            for h in range(self.attn_heads):
                attn_kernel = OverAll.init_emb(node_F, 1)
                self.attn_kernels[d].append([attn_kernel.to(device), attn_kernel.to(device)])

    def forward(self, inputs, rel_or_time=0):
        outputs = []
        outputs_time = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj_index = inputs[2]  # adj
        index = torch.tensor(adj_index, dtype=torch.int64)
        index = index.to(self.device)

        # adj = torch.sparse.FloatTensor(torch.LongTensor(index),
        #                                torch.FloatTensor(torch.ones_like(index[:,0])),
        #                                (self.node_size, self.node_size))
        sparse_indices = inputs[3]  # relation index  i.e. r_index
        sparse_val = inputs[4]  # relation value  i.e. r_val



        time_features = inputs[5]
        time_emb = inputs[6]

        t_sparse_indices = inputs[8]
        t_sparse_val = inputs[9]
        features = self.activation(features)
        outputs.append(features)


        time_features = self.activation(time_features)
        outputs_time.append(time_features)

        for l in range(self.depth):
            features_list = []
            time_features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]


                rels_sum = torch.sparse.FloatTensor(
                    torch.transpose(torch.LongTensor(sparse_indices), 0, 1),
                    torch.FloatTensor(sparse_val),
                    (self.triple_size, self.time_size)
                )  # relation matrix
                rels_sum = rels_sum.to(self.device)
                rels_sum = torch.matmul(rels_sum, rel_emb)
                neighs = features[index[:, 1]]
                # selfs = features[index[:, 0]]
                rels_sum = F.normalize(rels_sum, p=2, dim=1)
                neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum

                # Eq.3
                att1 = torch.squeeze(torch.matmul(rels_sum, attention_kernel[0]), dim=-1)
                att = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), att1, (self.node_size, self.node_size))
                # ??? dim ??


                times_sum = torch.sparse.FloatTensor(
                    torch.transpose(torch.LongTensor(t_sparse_indices), 0, 1),
                    torch.FloatTensor(t_sparse_val),
                    (self.triple_size, self.time_sizeT)
                )  # relation matrix
                times_sum = times_sum.to(self.device)
                times_sum = torch.matmul(times_sum, time_emb)
                time_neighs = time_features[index[:, 1]]
                # selfs = features[index[:, 0]]
                times_sum = F.normalize(times_sum, p=2, dim=1)
                time_neighs = time_neighs - 2 * torch.sum(time_neighs * times_sum, 1, keepdim=True) * times_sum

                # Eq.3
                time_att1 = torch.squeeze(torch.matmul(time_neighs, attention_kernel[1]), dim=-1)
                time_att = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), time_att1, (self.node_size, self.node_size))


                att = torch.sparse.softmax(att, dim=1)
                time_att = torch.sparse.softmax(time_att, dim=1)
                # ?
                # print(att1)
                # print(att.data)
                new_features = torch_scatter.scatter_add(
                    torch.transpose(neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), 0, 1),
                    index[:, 0])
                new_features = torch.transpose(new_features, 0, 1)
                features_list.append(new_features)


                new_features_time = torch_scatter.scatter_add(
                    torch.transpose(time_neighs * torch.unsqueeze(time_att.coalesce().values(), dim=-1), 0, 1),
                    index[:, 0])
                new_features_time = torch.transpose(new_features_time, 0, 1)
                time_features_list.append(new_features_time)

            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list)
                features = features * 2

                time_features = torch.cat(time_features_list)
                time_features = time_features * 2

            features = self.activation(features)
            outputs.append(features)

            time_features = self.activation(time_features)
            outputs_time.append(time_features)


        outputs = torch.cat(outputs, dim=1)
        # proxy_att = torch.matmul(F.normalize(outputs, dim=-1),
        #                          torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
        # proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
        # proxy_feature = outputs - torch.matmul(proxy_att, self.proxy)
        #
        # if self.use_bias:
        #     gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel) + self.bias)
        # else:
        #     gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))
        # outputs = gate_rate * outputs + (1 - gate_rate) * proxy_feature
        #
        #
        outputs_time = torch.cat(outputs_time, dim=1)
        # proxy_att = torch.matmul(F.normalize(outputs_time, dim=-1),
        #                          torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
        # proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
        # proxy_feature = outputs_time - torch.matmul(proxy_att, self.proxy)
        #
        # if self.use_bias:
        #     gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel2) + self.bias)
        # else:
        #     gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel2))
        #
        # outputs_time = gate_rate * outputs_time + (1 - gate_rate) * proxy_feature

        gate_rate = F.sigmoid(torch.matmul(outputs_time, self.gate_kernel3))
        outputs = gate_rate * outputs_time + (1 - gate_rate) * outputs

        #
        # gate_rate = F.sigmoid(torch.matmul(outputs, self.gate_kernel3))a
        # outputs = (1 - gate_rate) * outputs_time + (gate_rate) * outputs
        return outputs_time