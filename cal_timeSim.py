import gc

from config import *
import utils2
from utils import *
from models import *
from evaluate import evaluate
from tqdm import *
import torch
import torch.nn.functional as F
# import keras.backend as KTF
from seu_tkg import sinkhorn, cal_sims
from utils2 import test
from wl_test import getLast
import pandas as pd
import dto
import time


# import tensorflow as tf
# print(tf.test.is_gpu_available())
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# time_sim = multiple_sparse_ind(3, time_feature, dev_pair, sparse_rel_matrix)  # feature (85394,xxx)
def multiple_sparse_ind(depth, feature, test_pair, sparse_rel_matrix, right=None, time_no_agg=False):
    sims = cal_sims(test_pair, tf.cast(feature, tf.float32), right)
    if time_no_agg:
        return sims
    # multi layer
    for i in range(depth):
        feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, tf.cast(feature, tf.double))
        # print("sparse_rel_matrix.shape:", sparse_rel_matrix.shape)  # (85394, 85394)
        feature = tf.nn.l2_normalize(feature, axis=-1)
        # print("final feature: ", feature.shape)  # (85394, 14074)

        gc.collect()
        sims += cal_sims(test_pair, tf.cast(feature, tf.float32), right)
        print("sims.shape", sims.shape) # (29239, 29239)
    sims /= depth + 1
    return sims






import seu_tkg as seu

# time_feature = seu.get_feature(filename)   # year
time_feature_2 = seu.get_feature(filename, True) # day

# print(time_feature.shape)

time_suffix = str(which_file)
fn = 'time_sim' + time_suffix
fn2 = 'time_sim_day' + time_suffix

# if not unsup and train_ratio == 0.2:
#     fn += '_lessSeed'
# elif unsup:
#     fn += '_unsup'

train_pair, dev_pair, adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features, time_featuresT, t_indexT = load_data(
    filename, train_ratio=train_ratio, unsup=unsup, flag=1)
all_triples, node_size, _ = utils2.load_triples(filename, True)
# print("node_size: ", node_size)   #85394
sparse_rel_matrix = seu.construct_sparse_rel_matrix(all_triples, node_size)
# print("sparse_rel_matrix: ", sparse_rel_matrix.shape) # (85394, 85394)

tic = time.time()
# time_sim = multiple_sparse_ind(3, time_feature, dev_pair, sparse_rel_matrix)
time_sim_day = multiple_sparse_ind(3, time_feature_2, dev_pair, sparse_rel_matrix)
# print(time_sim_day.shape, time_sim_day)
toc = time.time()
time_encode_time = toc-tic
# dto.saveobj(time_sim, fn)
dto.saveobj(time_sim_day, fn2)

# if global_args.sep_eval:
#     time_sim2 = multiple_sparse_ind(2, time_feature, dev_pair2, sparse_rel_matrix)
#     dto.saveobj(time_sim2, fn + '2')
print('finish get time sim')


