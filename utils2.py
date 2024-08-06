import numba as nb
import numpy as np
import os
import time
import multiprocessing
import tensorflow as tf
from collections import defaultdict
import numpy as np
import torch
import math

multi = 946

def get_feature_sum(filename, i: int, shift_id=0, tf_idf=False, time_id=None, day=False,r_index=0):

    yearMap = get_timeMap(filename)
    monthDayMap = get_timeMap3(filename)
    from wl_test import getLast
    num_ent = getLast(filename + 'ent_ids_' + str(i))
    if i == 2:
        num_ent -= shift_id

    matrix_year = np.zeros((num_ent, 1))
    matrix_monthDay = np.zeros((num_ent, 1))


    for line in open(filename + 'triples_' + str(i), 'r'):
        words = line.split()

        head, r, tail, t1, t2 = [str(item) for item in words]
        head = int(head);r = int(r);tail = int(tail)

        a1 = 0
        b1 = 0
        a2 = 0
        b2 = 0


        if (len(t1) > 2):
            t_1 = t1.split('.')
            a1 = int(yearMap[t_1[0]])
            if (day):
                if (t_1[1] != '0101' and t_1[1] != '0100' and t_1[1] != '0000'):
                    a2 = int(monthDayMap[t_1[1]]) * multi + a1
                else:
                    a2 = 0


        if (len(t2) > 2):
            t_2 = t2.split('.')
            b1 = int(yearMap[t_2[0]])
            if (day):
                if (t_2[1] != '0101' and t_2[1] != '0100' and t_2[1] != '0000'):
                    b2 = int(monthDayMap[t_2[1]]) * multi + b1
                else:
                    b2 = 0


        t_encode1 = (a1 * 2000 + b1)
        t_encode2 = (a2 * 2000 + b2)

        if t_encode1 != 0 and r != 0:
            matrix_year[head - shift_id] += 1
            matrix_year[tail - shift_id] += 1
        if t_encode2 != 0 and r != 0:
            matrix_monthDay[head - shift_id] += 1
            matrix_monthDay[tail - shift_id] += 1

    return matrix_year, matrix_monthDay




#     m1 = get_feature_matrix(filename, 1, 0, TF, time_id)
def get_feature_matrix(filename, i: int, shift_id=0, tf_idf=False, time_id=None, day=False,r_index=0, matrix_year = None, matrix_monthDay = None):
    count = 0
    time_dict = dict()
    TS = 20000
    num_triple = 0
    time_set = set()
    entity_set = set()
    th = defaultdict(lambda: defaultdict(int))

    from wl_test import getLast
    num_ent = getLast(filename + 'ent_ids_' + str(i))

    if i == 2:
        num_ent -= shift_id

    matrix_time_cnt = np.zeros((num_ent, 1))

    yearMap = get_timeMap(filename)
    monthDayMap = get_timeMap2(filename)
    relationMap = get_relationMap(filename)
    flag = False

    for line in open(filename + 'triples_' + str(i), 'r'):
        num_triple += 1
        words = line.split()

        head, r, tail, t1, t2 = [str(item) for item in words]
        head = int(head);r = int(r);tail = int(tail)



        if int(relationMap.get(str(r), 0)) == r_index:
            flag = True
            a1 = 0
            b1 = 0

            if (len(t1) > 2):
                t_1 = t1.split('.')
                a1 = int(yearMap[t_1[0]])
                if (day):
                    if (t_1[1] != '0101' and t_1[1] != '0100' and t_1[1] != '0000'):
                        a1 = int(monthDayMap[t1]) * multi + a1
                    else:
                        a1 = 0


            if (len(t2) > 2):
                t_2 = t2.split('.')
                b1 = int(yearMap[t_2[0]])
                if (day):
                    if (t_2[1] != '0101' and t_2[1] != '0100' and t_2[1] != '0000'):
                        b1 = int(monthDayMap[t2]) * multi + b1
                    else:
                        b1 = 0


            t_encode1 = (a1 * 2000 + b1)
            if t_encode1 != 0:
                matrix_time_cnt[head - shift_id] += 1
                matrix_time_cnt[tail - shift_id] += 1
                if int(relationMap.get(str(r), 0)) != 0:
                    t_encode1 = t_encode1 + int(relationMap.get(str(r),0)) * 20000


            tt1 = time_id[t_encode1]

            # time_set.add(t)
            time_set.add(tt1)
            head, tail = head - shift_id, tail - shift_id
            # all the entity
            entity_set.add(head)
            entity_set.add(tail)



            # relation_index = int(relationMap.get(str(r), 0))
            # if relation_index != 0:
            #     relation_index -= 893405

            if t_encode1 > 0:
                th[head][tt1] += 1
                th[tail][tt1 + TS] += 1




    index, value = [], []
    # num_ent = len(entity_set)



    if time_id is not None:
        num_time = len(time_id.keys())
    else:
        num_time = len(time_set)


    for ent, dic in th.items():
        for time, cnt in dic.items():
            t = time if time < TS else time + num_time - TS  # different id for head and tail
            index.append((ent, t))
            value.append(cnt)


    index = torch.LongTensor(index)
    # print(num_ent, num_time)

    matrix = torch.sparse_coo_tensor(torch.transpose(index, 0, 1), torch.Tensor(value),
                                     (num_ent, 2 * num_time))
    print(r_index)
    print(flag)

    # if (day):
    #     if (r_index != -1):
    #         percentage_matrix = np.ones((num_ent, 1))
    #     else:
    #         matrix_monthDay[matrix_monthDay == 0] = 1e-10
    #         percentage_matrix =  matrix_monthDay / (matrix_monthDay - matrix_time_cnt)
    # else:
    #     if (r_index == 0):
    #         percentage_matrix = np.ones((num_ent, 1))
    #     else:
    #         matrix_year[matrix_year == 0] = 1e-10
    #         percentage_matrix = matrix_year / (matrix_year - matrix_time_cnt)
    #
    #
    #
    #
    # matrix = torch.sparse_coo_tensor(torch.transpose(index, 0, 1), torch.Tensor(value),(num_ent, 2 * num_time))
    # # print("num_ent", num_ent, matrix.size())
    #
    # weighted_values = value * percentage_matrix[index[:, 0]].squeeze()
    # weighted_matrix = torch.sparse_coo_tensor(torch.transpose(index, 0, 1), weighted_values, (num_ent, 2 * num_time))
    # 打印结果的形状
    # print("加权后的稀疏张量形状：", weighted_matrix.size())

    # if r_index == 0:
    #     return weighted_matrix, matrix_time_cnt

    return matrix



def get_link(filename, shift_id=0):
    links = []
    for line in open(filename + 'sup_pairs', 'r'):
        e1, e2 = line.split()
        links.append((int(e1), int(e2) - shift_id))
    for line in open(filename + 'ref_pairs', 'r'):
        e1, e2 = line.split()
        links.append((int(e1), int(e2) - shift_id))
    return links

def get_timeMap(filename):
    timeMap = {}
    for line in open(filename + 'time_id', 'r'):
        # print(line)
        words = line.split()
        # print(len(words))
        value, key = [str(item) for item in words]
        timeMap[key] = value
    return timeMap

def get_timeMap2(filename):
    timeMap = {}
    for line in open(filename + 'time_id2', 'r'):
        # print(line)
        words = line.split()
        # print(len(words))
        value, key = [str(item) for item in words]
        timeMap[key] = value
    return timeMap

def get_timeMap3(filename):
    timeMap = {}
    for line in open(filename + 'time_id3', 'r'):
        # print(line)
        words = line.split()
        # print(len(words))
        value, key = [str(item) for item in words]
        timeMap[key] = value
    return timeMap

def get_relationMap(filename):
    relationMap = {}
    for line in open(filename + 'rels_same', 'r'):
        # print(line)
        words = line.split()
        # print(len(words))
        key, value = [str(item) for item in words]
        relationMap[key] = value
    return relationMap

# time_id = get_time(filename)
def get_time(filename, day=False):
    yearMap = get_timeMap(filename)
    monthDayMap = get_timeMap2(filename)
    relationMap = get_relationMap(filename)

    time_dict = dict()
    count = 0
    for i in [0, 1]:
        for line in open(filename + 'triples_' + str(i + 1), 'r'):
            words = line.split()
            head, r, tail, t1, t2 = [str(item) for item in words]
            head = int(head);r = int(r);tail = int(tail)
            a1 = 0
            b1 = 0

            if (len(t1) > 2):
                t_1 = t1.split('.')
                a1 = int(yearMap[t_1[0]])
                if (day):
                    if (t_1[1] != '0101' and t_1[1] != '0100' and t_1[1] != '0000'):
                        a1 = int(monthDayMap[t1]) * multi + a1
                    else:
                        a1 = 0

            if (len(t2) > 2):
                t_2 = t2.split('.')
                b1 = int(yearMap[t_2[0]])
                if (day):
                    if (t_2[1] != '0101' and t_2[1] != '0100' and t_2[1] != '0000'):
                        b1 = int(monthDayMap[t2]) * multi + b1
                    else:
                        b1 = 0


            tt1 = (a1 * 2000 + b1)

            if (tt1 != 0 and int(relationMap.get(str(r),0)) != 0):
                tt1 = tt1 + int(relationMap.get(str(r),0)) * 20000
            # print(relationMap.get(r,0))
            # give each time a number
            if tt1 not in time_dict.keys():
                time_dict[tt1] = count
                count += 1

    return time_dict


def load_triples(file_path, reverse=True):
    @nb.njit
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            if reverse:
                reversed_triples[i, 1] = triples[i, 1] + rel_size
            else:
                reversed_triples[i, 1] = triples[i, 1]
        return reversed_triples

    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()

    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()

    triples = np.array([line.replace("\n","").split("\t")[0:3] for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:, 0]), np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:, 1]) + 1

    all_triples = np.concatenate([triples, reverse_triples(triples)], axis=0)
    all_triples = np.unique(all_triples, axis=0)

    return all_triples, node_size, rel_size*2 if reverse else rel_size

def load_aligned_pair(file_path,ratio = 0.3):
    if "sup_ent_ids" not in os.listdir(file_path):
        with open(file_path + "ref_ent_ids") as f:
            aligned = f.readlines()
    else:
        with open(file_path + "ref_ent_ids") as f:
            ref = f.readlines()
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
        aligned = ref + sup

    aligned = np.array([line.replace("\n", "").split("\t") for line in aligned]).astype(np.int64)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]

def test(sims,mode = "sinkhorn", batch_size = 1024):
    if mode == "sinkhorn":
        results = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch*batch_size:(epoch+1)*batch_size]
            rank = tf.argsort(-sim, axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(sims)))])
            x = np.expand_dims(ans_rank, axis=1)
            y = tf.tile(x, [1, len(sims)])
            results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank, axis=1), [1, len(sims)]))).numpy())
        results = np.concatenate(results, axis=0)


        @nb.jit(nopython=True)
        def cal(results):
            hits1, hits10, mrr = 0, 0, 0
            for x in results[:, 1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1, hits10, mrr
        hits1, hits10, mrr = cal(results)
        print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(sims)*100, hits10/len(sims)*100, mrr/len(sims)*100))
        return hits1/len(sims), hits10/len(sims), mrr/len(sims)
    else:
        c = 0
        for i, j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%" %(100 * c/len(sims[0])))
        return c/len(sims[0])


def find_pairs(test_pair, sim1, sim2):
    rank = tf.argmax(sim1, axis=-1)
    left_id = test_pair[:, 0]
    right_id = test_pair[:, 1]
    cnt = 0
    link_1, link_2 = set(), set()
    for x in range(len(rank)):
        link_1.add((left_id[x], right_id[rank[x]]))
    rank2 = tf.argmax(sim2, axis=-1)
    for x in range(len(rank2)):
        link_2.add((left_id[rank2[x]], right_id[x]))
    overall = link_1.intersection(link_2)
    print(len(overall))
    test_set = set(zip(left_id, right_id))

    print(len(overall.intersection(test_set)))
    y = len(overall.intersection(test_set))
    print(y/len(overall))
    links = list(overall)

    with open('unsup_link', 'w') as f:
        for link in links:
            f.write(str(link[0]) + '\t' + str(link[1]) + '\n')
    return links