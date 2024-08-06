from tqdm import tqdm
from scipy import optimize
import tensorflow as tf
from utils2 import *
import json
import os
from config import filename

seed = 12345
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def cal_sims(test_pair, feature, right=None):
    if right is None:
        print("feature: ",feature.shape)
        feature_a = tf.gather(indices=test_pair[:, 0], params=feature)
        feature_b = tf.gather(indices=test_pair[:, 1], params=feature)
    else:
        feature_a = feature[:right]
        feature_b = feature[right:]
    fb = tf.transpose(feature_b, [1, 0])
    return tf.matmul(feature_a, fb)


# Hungarian algorithm, only for the CPU
# result = optimize.linear_sum_assignment(sims, maximize=True)
# test(result, "hungarian")

# Sinkhorn operation
def sinkhorn(sims, eps=1e-6):
    sims = tf.exp(sims * 50)
    for k in range(10):
        sims = sims / (tf.reduce_sum(sims, axis=1, keepdims=True) + eps)
        sims = sims / (tf.reduce_sum(sims, axis=0, keepdims=True) + eps)
    return sims

def construct_sparse_rel_matrix(all_triples, node_size):
    dr = {}
    for x, r, y in all_triples:
        if r not in dr:
            dr[r] = 0
        dr[r] += 1
    sparse_rel_matrix = []
    for i in range(node_size):
        sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)])
    for h, r, t in all_triples:
        sparse_rel_matrix.append([h, t, np.log(len(all_triples) / dr[r])])
    sparse_rel_matrix = np.array(sorted(sparse_rel_matrix, key=lambda x: x[0]))
    sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:, :2], values=sparse_rel_matrix[:, 2],
                                        dense_shape=(node_size, node_size))
    return sparse_rel_matrix


def get_feature(filename, day=False):
    global m1, m2
    if filename == 'data/ICEWS05-15/':
        shift = 9517
        overlap = None
        time_id = None
    elif filename == 'data/YAGO-WIKI50K/':
        shift = 42666 #49629
        time_id = get_time(filename, day) # get the whole time id(different from the file"time_id")
        print("len(time_id): ", len(time_id))
        # print(len(time_id))
    else:
        shift = 19493
        time_id = get_time(filename)
        print(len(time_id))
    TF = False

    matrix_year, matrix_monthDay = get_feature_sum(filename, 1, 0, TF, time_id,day)
    matrix_year2, matrix_monthDay2 = get_feature_sum(filename, 2, shift, TF, time_id, day)


    m1 = get_feature_matrix(filename, 1, 0, TF, time_id,day, 0, matrix_year, matrix_monthDay)

    m2 = get_feature_matrix(filename, 2, shift, TF, time_id,day, 0, matrix_year2, matrix_monthDay2)
    # print("yes2",m2)


    for r in range(1,11):
        TF = False
        m1 += get_feature_matrix(filename, 1, 0, TF, time_id,day,r,matrix_year, matrix_monthDay)
        m2 += get_feature_matrix(filename, 2, shift, TF, time_id,day,r,matrix_year2, matrix_monthDay2)



    # print("m1_2", m1.size())
    # print("m2_2", m2.size())

    if not TF:
        from config import global_args
        if not global_args.time_no_agg:
            # m1, m2 = torch.softmax(m1, dim=1), torch.softmax(m2, dim=1)

            m1, m2 = torch.sparse.softmax(m1, dim=1), torch.sparse.softmax(m2, dim=1)
    feature = torch.vstack((m1.to_dense(), m2.to_dense())).numpy()

    # print("m1", m1.size())
    # print("m2", m2.size())
    # feature = torch.vstack((m1, m2)).numpy()

    feature = tf.nn.l2_normalize(feature, axis=-1)
    feature = tf.cast(feature, tf.float64)
    # print("feature: ", feature.shape)

    return feature


def main():
    # choose the GPU, "-1" represents using the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    file_path = filename
    all_triples, node_size, rel_size = load_triples(file_path, True)
    # train_pair, test_pair = load_aligned_pair(file_path, ratio=0)
    test_pair = get_link(filename)
    test_pair_re = [(a, b) for (b, a) in test_pair]
    test_pair = np.array(test_pair)
    test_pair_re = np.array(test_pair_re)

    # build the relational adjacency matrix
    sparse_rel_matrix = construct_sparse_rel_matrix(all_triples, node_size)
    feature = get_feature(filename)

    # choose the graph depth L and feature propagation
    depth = 2
    sims = cal_sims(test_pair, feature)
    sims2 = cal_sims(test_pair_re, feature)
    for i in range(depth):
        feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, feature)
        feature = tf.nn.l2_normalize(feature, axis=-1)
        sims += cal_sims(test_pair, feature)
        # sims2 += cal_sims(test_pair_re, feature)
    sims /= depth + 1
    # sims2 /= depth + 1

    sims = sinkhorn(sims)
    # sims2 = sinkhorn(sims2)

    test(sims, "sinkhorn")
    # print('------------ begin find pairs ---------')
    # find_pairs(test_pair, sims, sims2)
    # print('------------- end find pairs ----------')

    links = [(i, i) for i in range(len(sims))]
    links = torch.LongTensor(links)
    print('lxz version')
    evaluate_sim_matrix(torch.transpose(links, 0, 1), torch.Tensor(sims.numpy()))


if __name__ == '__main__':
    main()
