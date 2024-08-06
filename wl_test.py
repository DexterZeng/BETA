import math

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import networkx as nx
import numpy as np
from config import filename
from config import which_file
import torch
from torch import Tensor
# filename = 'data/ICEWS05-15/'
# subtree_height = 8
from tqdm import tqdm, trange


def ind2sparse(indices: Tensor, size, size2=None, dtype=torch.float, values=None):
    device = indices.device
    if isinstance(size, int):
        size = (size, size if size2 is None else size2)

    assert indices.dim() == 2 and len(size) == indices.size(0)
    if values is None:
        values = torch.ones([indices.size(1)], device=device, dtype=dtype)
    else:
        assert values.dim() == 1 and values.size(0) == indices.size(1)
    return torch.sparse_coo_tensor(indices, values, size)


def read_unsup_link(unsup=True):
    link_x2y = {}
    link_y2x = {}
    with open(filename + ('unsup_link' if unsup else 'sup_pairs'), 'r') as f:
        from config import global_args
        lines = f.readlines()
        if global_args.train_ratio == 20:
            lines = lines[:int(len(lines) * 0.2)]
        for line in lines:
            a, b = line.strip().split()
            link_x2y[int(a)] = int(b)
            link_y2x[int(b)] = int(a)
    return link_x2y, link_y2x


# sup_links = read_unsup_link(False)
# unsup_links = read_unsup_link()


def build_new_triples(shift_id, i, links):
    index2 = set()
    for line in open(filename + 'triples_' + str(3 - i), 'r'):
        words = line.split()
        if len(words) == 4:
            head, r, tail, t = [int(item) for item in words]
        else:
            head, r, tail, t1, t2 = [int(item) for item in words]
        current_link = links[2 - i]
        if head in current_link and tail in current_link:
            head, tail = current_link[head] - shift_id, current_link[tail] - shift_id
            index2.add((head, tail))
            index2.add((tail, head))
    return index2


def getAdjMat3(i, known_labels, return_g=True):
    index = set()
    shift_id = 0
    if i == 2:
        shift_id = getLast(filename + 'ent_ids_1')
    ent_set = set()
    # unsup_index = build_new_triples(shift_id, i, unsup_links)
    # sup_index = build_new_triples(shift_id, i, sup_links)
    # import random
    # random.choices()
    # new_index = unsup_index
    # print(len(unsup_index), len(sup_index), len(new_index))
    # index = new_index
    for line in open(filename + 'triples_' + str(i), 'r'):
        words = line.split()
        if len(words) == 4:
            head, r, tail, t = [int(item) for item in words]
        else:
            head, r, tail, t1, t2 = [int(item) for item in words]

        head, tail = head - shift_id, tail - shift_id
        ent_set.add(head), ent_set.add(tail)
        index.add((head, tail))
        index.add((tail, head))
    # num_ent = len(ent_set)
    num_ent = getLast(filename + 'ent_ids_' + str(i))
    if i == 2:
        num_ent -= shift_id

    for i in range(num_ent):
        index.add((i, i))
    if return_g:
        g = nx.Graph()
        labels = {}
        for (s, t) in index:
            g.add_edge(s, t)
            if s not in labels:
                labels[s] = s + 100000

            if t not in labels:
                labels[t] = t + 100000
        for k, v in known_labels.items():
            labels[k] = v
        nx.set_node_attributes(g, labels, 'label')
        return g
    else:
        return ind2sparse(torch.tensor(list(index)).t(), size=(num_ent, num_ent))


def getAdjMat(i, known_labels, return_g=True):
    index = []
    shift_id = 0
    if i == 2:
        shift_id = getLast(filename + 'ent_ids_1')
    ent_set = set()
    for line in open(filename + 'triples_' + str(i), 'r'):
        words = line.split()
        if len(words) == 4:
            head, r, tail, t = [int(item) for item in words]
        else:
            head, r, tail, t1, t2 = [str(item) for item in words]
            head = int(head);r = int(r);tail = int(tail)
        head, tail = head - shift_id, tail - shift_id
        ent_set.add(head), ent_set.add(tail)
        index.append((head, tail))
        index.append((tail, head))
    # num_ent = len(ent_set)
    num_ent = getLast(filename + 'ent_ids_' + str(i))
    if i == 2:
        num_ent -= shift_id
    for i in range(num_ent):
        index.append((i, i))
    if return_g:
        g = nx.Graph()
        labels = {}
        for (s, t) in index:
            g.add_edge(s, t)
            if s not in labels:
                labels[s] = s + 100000
            if t not in labels:
                labels[t] = t + 100000
        for k, v in known_labels.items():
            labels[k] = v
        nx.set_node_attributes(g, labels, 'label')
        return g
    else:
        return ind2sparse(torch.tensor(index).t(), size=(num_ent, num_ent))


def getAdjMat2(i, *args, **kwargs):
    all_triple = []
    dr = {}
    shift_id = 0
    index = []
    value = []
    if i == 2:
        shift_id = getLast(filename + 'ent_ids_1')
    ent_set = set()
    for line in open(filename + 'triples_' + str(i), 'r'):
        words = line.split()
        if len(words) == 4:
            head, r, tail, t = [int(item) for item in words]
        else:
            head, r, tail, t1, t2 = [int(item) for item in words]
        if r not in dr:
            dr[r] = 0
        dr[r] += 1
        head, tail = head - shift_id, tail - shift_id
        ent_set.add(head), ent_set.add(tail)
        all_triple.append((head, r, tail))
    num_ent = len(ent_set)
    for i in range(num_ent):
        index.append((i, i))
        value.append(1 / num_ent)
    for h, r, t in all_triple:
        index.append((h, t))
        value.append(1 / dr[r])
    return ind2sparse(torch.tensor(index).t(), size=(num_ent, num_ent), values=torch.tensor(value).to(torch.float))


def getLast(filename, pos=0):
    f = open(filename, 'r', encoding='utf-8')
    last = f.readlines()[-1]
    last = int(last.split()[pos])
    return last + 1


t_dic = dict()


def get_tid(val):
    if val not in t_dic:
        t_dic[val] = len(t_dic)
    return t_dic[val]


def get_time_shift():
    if which_file == 0:
        shift_id = 4017
    elif which_file == 1:
        shift_id = 2896
    else:
        shift_id = 2383
    return shift_id


def getTimeMat(i, return_g=True):
    num_ent = getLast(filename + 'ent_ids_{}'.format(i))
    index = []
    shift_id = 0
    if i == 2:
        shift_id = getLast(filename + 'ent_ids_1')
    ent_set = set()
    from collections import defaultdict
    num_of_time = defaultdict(int)
    for line in open(filename + 'triples_' + str(i), 'r'):
        words = line.split()
        if len(words) == 4:
            head, r, tail, t = [int(item) for item in words]
        else:
            head, r, tail, t1, t2 = [int(item) for item in words]
            t = t1 * 1000 + t2
            t = get_tid(t)
        head, tail = head - shift_id, tail - shift_id
        ent_set.add(head), ent_set.add(tail)

        index.append((head, t + num_ent - shift_id))
        index.append((tail, t + num_ent - shift_id + get_time_shift()))
        # index.append((tail, t + num_ent - shift_id))
        num_of_time[t] += 1
        num_of_time[t + get_time_shift()] += 1
    if return_g:
        g = nx.Graph()
        labels = {}
        for (s, t) in index:
            g.add_edge(s, t)
            if s not in labels:
                labels[s] = s

            if t not in labels:
                labels[t] = t
                if i == 2:
                    labels[t] += 10000000
        nx.set_node_attributes(g, labels, 'label')

        return g
    else:
        index = [(h, t - num_ent + shift_id) for (h, t) in index]
        # todo
        value = [1 / num_of_time[t] for (_, t) in index]
        t_len = max([t for (h, t) in index])
        return torch.tensor(index).t(), num_ent - shift_id, t_len + 1, value


class WLKernel:
    def __init__(self, train_pair, wl_height=8):
        train_pair = np.copy(train_pair)
        left = train_pair[:, 0]
        right = train_pair[:, 1]
        right -= getLast(filename + 'ent_ids_1')
        label_left, label_right = {}, {}
        for i, l in enumerate(left):
            label_left[l] = i
        for i, r in enumerate(right):
            label_right[r] = i
        self.rels = [getAdjMat(1, label_left), getAdjMat(2, label_right)]
        self.times = [getTimeMat(1), getTimeMat(2)]
        self.wl_height = wl_height

        self.rel_adjs = [getAdjMat3(1, label_left, False), getAdjMat3(2, label_right, False)]

        self.time_adjs = []
        sp_info = []
        time_len = 0
        for i in range(1, 3):
            sp_id, num_ent, t_len, value = getTimeMat(i, False)
            sp_info.append((sp_id, num_ent, value))
            # sp_info.append((sp_id, num_ent))
            time_len = max(time_len, t_len)

        self.time_adjs = [ind2sparse(sp_id, (num_ent, time_len), values=torch.tensor(value)) for (sp_id, num_ent, value)
                          in sp_info]

    @torch.no_grad()
    def adj_sim(self, P: np.ndarray) -> float:
        P = torch.from_numpy(P).cuda()
        if P.size(0) < 20000:
            return self.sparse_adj_sim(P)
        left = torch.sparse.mm(self.rel_adjs[0].cuda(), P)
        right = torch.sparse.mm(self.rel_adjs[1].cuda().t(),
                                P.t()).t()
        rel_sim = torch.norm(left - right).item()
        print('REL sim is', rel_sim, self.rl() * rel_sim)
        time_sim = torch.norm(
            self.time_adjs[0].cuda() - torch.sparse.mm(self.time_adjs[1].cuda().t(), P.t()).t()).item()
        print('TIME sim is', time_sim, self.tm() * time_sim)
        print('Final Sim is ', self.rl() * rel_sim + self.tm() * time_sim)
        return self.rl() * rel_sim + self.tm() * time_sim

    @torch.no_grad()
    def sparse_adj_sim(self, P: np.ndarray):
        P = torch.from_numpy(P)
        import utils_large as ul
        # P = ul.dense_to_sparse_mini_batch(P).cuda()
        if P.size(0) > 20000:
            # to save time
            P = ul.remain_topk_sim(P, k=1)
            P = ind2sparse(P._indices(), P.size()).coalesce().cuda()
        else:
            P = ul.remain_topk_sim(P, k=1)
            P = ind2sparse(P._indices(), P.size()).coalesce().cuda()
            # P = ul.remain_topk_sim(P, k=1).coalesce().cuda()
            # P = ul.remain_topk_sim(P, k=500).coalesce().cuda()
        rel_sim = torch.norm((
                ul.spspmm(self.rel_adjs[0].cuda(), P).cpu() -
                ul.spspmm(self.rel_adjs[1].cuda().t(), P.t()).t().cpu())).item()
        print('REL sim is', rel_sim, self.rl() * rel_sim)
        time_sim = torch.norm(
            self.time_adjs[0] - ul.spspmm(self.time_adjs[1].cuda().t(), P.t()).t().cpu()).item()
        print('TIME sim is', time_sim, self.tm() * time_sim)
        print('Final Sim is ', self.rl() * rel_sim + self.tm() * time_sim)
        return self.rl() * rel_sim + self.tm() * time_sim, rel_sim, time_sim

    def tm(self):
        return self.time_kernel[0][1]

    def rl(self):
        return self.rel_kernel[0][1]

    def calGraphKernels(self, wl_height):
        rel_kernel = None
        time_kernel = None
        gk = WeisfeilerLehman(n_iter=wl_height, base_graph_kernel=VertexHistogram, normalize=True, verbose=True)
        self.rel_graph = graph_from_networkx(self.rels, node_labels_tag='label')
        self.time_graph = graph_from_networkx(self.times, node_labels_tag='label')
        # self.time_graph = graph_from_networkx(self.times)
        rel_kernel = gk.fit_transform(self.rel_graph)
        gk = WeisfeilerLehman(n_iter=wl_height, base_graph_kernel=VertexHistogram, normalize=True, verbose=True)
        time_kernel = gk.fit_transform(self.time_graph)
        self.rel_kernel, self.time_kernel = rel_kernel, time_kernel
        print('WL kernel complete', self.tm(), self.rl())

