import numpy as np
import scipy.sparse as sp
import scipy
import tensorflow as tf
import os
import multiprocessing
from torch import Tensor
import torch


def view2(x):
    if x.dim() == 2:
        return x
    return x.view(-1, x.size(-1))


def view3(x: Tensor) -> Tensor:
    if x.dim() == 3:
        return x
    return x.view(1, x.size(0), -1)


def view_back(M):
    return view3(M) if M.dim() == 2 else view2(M)


def cosine_sim(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

from utils2 import get_timeMap, get_timeMap2, get_relationMap


def get_time_split(t, yearMap, monthDayMap, r = 0):
    relationMap = get_relationMap("data/YAGO-WIKI50K/")

    a1 = 0
    a2 = 0

    if (len(t) > 2):
        t_1 = t.split(".")
        a1 = int(yearMap[t_1[0]])
        if (t_1[1] != '0101' and t_1[1] != '0100' and t_1[1] != '0000'):
            # a2 = int(monthDayMap[t_1[1]]) * 960 + a1
            a2 = int(monthDayMap[t_1[1]])
        # if (int(relationMap.get(str(r), 0)) != 0):
        #     a1 = a1 + (int(relationMap.get(str(r), 0)) - 893406 + 1326) * 946
        #
        #     if a2 > 0:
        #         a2 = a2  +  int(relationMap.get(str(r), 0))  # initial a2 is unique
    # else:
        # a1 =  int(relationMap.get(str(r), 0))

    return  a1, a2

def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    time = set([0])
    yearMap = get_timeMap("data/YAGO-WIKI50K/")
    monthDayMap = get_timeMap2("data/YAGO-WIKI50K/")
    relationMap = get_relationMap("data/YAGO-WIKI50K/")


    for line in open(file_name, 'r'):
        para = line.split()
        if len(para) == 5:
            head, r, tail, ts, te = [str(item) for item in para]
            head = int(head)
            tail = int(tail)
            r = int(r)

            entity.add(head)
            entity.add(tail)
            rel.add(r + 1)
            # time.add(ts + 1);
            # time.add(te + 1)
            # triples.append((head, r + 1, tail, ts + 1, te + 1))

            ts1_1, ts1_2 = get_time_split(ts, yearMap, monthDayMap, r)
            te1_1, te1_2 = get_time_split(te, yearMap, monthDayMap, r)

            time.add(ts1_1 + 1)
            if ts1_2 != 0:
                time.add(ts1_2 + 1)

            time.add(te1_1 + 1)
            if te1_2 != 0:
                time.add(te1_2 + 1)


            triples.append((head, r+1, tail, ts1_1 + 1, te1_1 + 1))

            if ts1_2 != 0 or te1_2 != 0:
                triples.append((head, r+1, tail, ts1_2 + 1, te1_2 + 1))

        else:
            head, r, tail, t = [int(item) for item in para]
            entity.add(head);
            entity.add(tail);
            rel.add(r + 1)
            time.add(t + 1)
            triples.append((head, r + 1, tail, t + 1))
    return entity, rel, triples, time


def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair


def get_matrix(triples, entity, rel, time):
    ent_size = max(entity) + 1
    rel_size = (max(rel) + 1)
    time_size = (max(time) + 1)
    print(ent_size, rel_size, time_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    time_link = sp.lil_matrix((ent_size, time_size))
    #time_link = np.zeros((ent_size, time_size), dtype=int)  # new adding

    for i in range(max(entity) + 1):
        adj_features[i, i] = 1

    # 先进行判断，说明数据集中要么都是时间点，要么都是区间，后续可能需要改
    if len(triples[0]) < 5:
        for h, r, t, tau in triples:
            adj_matrix[h, t] = 1;
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1;
            adj_features[t, h] = 1
            radj.append([h, t, r, tau]);
            radj.append([t, h, r + rel_size, tau])
            time_link[h][tau] += 1;
            time_link[t][tau] += 1
            rel_out[h][r] += 1;
            rel_in[t][r] += 1
    else:
        for h, r, t, ts, te in triples:
            adj_matrix[h, t] = 1;
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1;
            adj_features[t, h] = 1
            radj.append([h, t, r, ts]);
            radj.append([h, t, r + rel_size, te])
            # time_link[h][te] += 1;
            # time_link[h][ts] += 1
            # time_link[t][ts] += 1;
            # time_link[t][te] += 1
            time_link[h,te] += 1;
            time_link[h,ts] += 1
            time_link[t,ts] += 1;
            time_link[t,te] += 1
            rel_out[h][r] += 1;
            rel_in[t][r] += 1
    count = -1
    s = set()
    d = {}
    r_index, t_index, r_val = [], [], []
    for h, t, r, tau in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            t_index.append([count, tau])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            t_index.append([count, tau])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    time_features = time_link
    time_features = normalize_adj(sp.lil_matrix(time_features))

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))
    return adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features


def load_data(lang, train_ratio=0.3, unsup=False):
    # 加载三元组所有信息
    entity1, rel1, triples1, time1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2, time2 = load_triples(lang + 'triples_2')
    # modified here #

    train_pair = load_alignment_pair(lang + 'sup_pairs')
    dev_pair = load_alignment_pair(lang + 'ref_pairs')
    if train_ratio < 0.25:
        train_ratio = int(3103)
        dev_pair = train_pair[train_ratio:] + dev_pair
        train_pair = train_pair[:train_ratio]
        print(len(train_pair))
    if unsup:
        dev_pair = train_pair + dev_pair
        train_pair = load_alignment_pair(lang + 'unsup_link')

    adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_feature = \
        get_matrix(triples1 + triples2, entity1.union(entity2), rel1.union(rel2), time1.union(time2))

    return np.array(train_pair), np.array(dev_pair), adj_matrix, np.array(r_index), np.array(r_val), \
           np.array(t_index), adj_features, rel_features, time_feature
