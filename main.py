import gc

import numpy as np
from config import *
import utils2
from utils import *
from models import *
from evaluate import evaluate
from tqdm import *
import torch
import torch.nn.functional as F
import keras.backend as KTF
from seu_tkg import sinkhorn, cal_sims
from utils2 import test
from wl_test import getLast
import pandas as pd
import dto
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

import seu_tkg as seu

train_pair, dev_pair, adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features, time_featuresT, t_indexT = load_data(
    filename, train_ratio=train_ratio, unsup=unsup)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
time_matrixT, time_valT = np.stack(time_featuresT.nonzero(), axis=1), time_featuresT.data

if global_args.sep_eval:
    if global_args.unsup:
        addition = '2'
    elif global_args.train_ratio == 20:
        addition = '1'
    else:
        addition = ''
    dev_pair = load_alignment_pair(filename + 'time_sensitive_link' + addition)
    dev_pair = np.array(dev_pair)
    dev_pair2 = load_alignment_pair(filename + 'not_sensitive_link' + addition)
    dev_pair2 = np.array(dev_pair2)

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
time_size = time_features.shape[1]
time_sizeT = time_featuresT.shape[1]

triple_size = len(adj_matrix)  # not triple size, but number of diff(h, t)
eval_epoch = 3
node_hidden = 100  # 128
rel_hidden = 100
batch_size = 512
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2
torch.cuda.set_device(2)
device = 'cuda'

training_time = 0.
grid_search_time = 0.
time_encode_time = 0.

import dto
from wl_test import WLKernel


# wl_kernel = WLKernel(train_pair)
# wl_kernel.calGraphKernels(8)


def get_embedding(index_a, index_b, vec):
    vec = vec.detach().numpy()
    Lvec = np.array([vec[e] for e in index_a])
    Rvec = np.array([vec[e] for e in index_b])
    Lvec = Lvec / (np.linalg.norm(Lvec, axis=-1, keepdims=True) + 1e-5)
    Rvec = Rvec / (np.linalg.norm(Rvec, axis=-1, keepdims=True) + 1e-5)
    return Lvec, Rvec


def multiple_sparse_ind(depth, feature, test_pair, sparse_rel_matrix, right=None, time_no_agg=False):
    sims = cal_sims(test_pair, tf.cast(feature, tf.float32), right)
    if time_no_agg:
        return sims
    for i in range(depth):
        feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, tf.cast(feature, tf.double))
        feature = tf.nn.l2_normalize(feature, axis=-1)
        sims += cal_sims(test_pair, tf.cast(feature, tf.float32), right)
    sims /= depth + 1
    return sims


def layer_normalization(embedding):
    mean = embedding.mean(dim=1, keepdim=True)
    variance = embedding.var(dim=1, keepdim=True)
    normalized_embedding = (embedding - mean) / torch.sqrt(variance + 1e-8)
    gamma = torch.ones_like(mean)
    beta = torch.zeros_like(mean)
    return gamma * normalized_embedding + beta


class CrossBatchNormalization(torch.nn.Module):
    def __init__(self, num_features, cbn_smoothing=0.9):
        super(CrossBatchNormalization, self).__init__()
        self.num_features = num_features
        self.cbn_smoothing = cbn_smoothing
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        device = x.device
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        if self.training:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)

            self.running_mean = self.cbn_smoothing * self.running_mean + (1 - self.cbn_smoothing) * batch_mean
            self.running_var = self.cbn_smoothing * self.running_var + (1 - self.cbn_smoothing) * batch_var
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + 1e-5)
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return x_normalized

def align_loss(align_input, embedding, cbn_smoothing=0.8):
    def squared_dist(x):
        A, B = x
        row_norms_A = torch.sum(torch.square(A), dim=1)
        row_norms_A = torch.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = torch.sum(torch.square(B), dim=1)
        row_norms_B = torch.reshape(row_norms_B, [1, -1])  # Row vector.
        # may not work
        return row_norms_A + row_norms_B - 2 * torch.matmul(A, torch.transpose(B, 0, 1))

    # modified
    left = torch.tensor(align_input[:, 0])
    right = torch.tensor(align_input[:, 1])
    l_emb = embedding[left]
    r_emb = embedding[right]

    # l_emb = cbn(l_emb.to(device))
    # r_emb = cbn(r_emb.to(device))

    # l_emb = layer_normalization(l_emb)
    # r_emb = layer_normalization(r_emb)

    pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
    r_neg_dis = squared_dist([r_emb, embedding])
    l_neg_dis = squared_dist([l_emb, embedding])

    l_loss = pos_dis - l_neg_dis + gamma
    l_loss = l_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)

    r_loss = pos_dis - r_neg_dis + gamma
    r_loss = r_loss * (1 - F.one_hot(left, num_classes=node_size) - F.one_hot(right, num_classes=node_size)).to(device)


    lamb, tau = 30, 10
    l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
    r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
    return torch.mean(l_loss + r_loss)


def save_suffix(save_final=False, cnt_call=1):
    suf = str(which_file) + '_' + str(len(train_pair))
    # dual_no_time = global_args.dual_no_time
    if global_args.dual_no_time:
        suf += '_dual_no_time'
    if save_final:
        if global_args.sep_eval:
            suf += '_sensitive' + str(cnt_call)
        if global_args.time_no_agg:
            suf += '_time_no_agg'
    return suf
    # no_sinkhorn = global_args.no_sinkhorn
    # no_time_feature = global_args.no_time_feature
    # no_rel_feature = global_args.no_rel_feature


def train():
    print('begin')
    # inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    model = OverAll(node_size=node_size, node_hidden=node_hidden,
                    rel_size=rel_size, rel_hidden=rel_hidden,
                    time_size=time_size, time_sizeT=time_sizeT,
                    rel_matrix=rel_matrix, ent_matrix=ent_matrix,
                    time_matrix=time_matrix, time_matrixT=time_matrixT,
                    triple_size=triple_size, dropout_rate=dropout_rate,
                    depth=depth, device=device)
    model = model.to(device)
    # opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0)
    print('model constructed')

    evaluater = evaluate(dev_pair)
    rest_set_1 = [e1 for e1, e2 in dev_pair]
    rest_set_2 = [e2 for e1, e2 in dev_pair]
    np.random.shuffle(rest_set_1)
    np.random.shuffle(rest_set_2)

    epoch = 10 if train_ratio > 0.2 else 20
    if unsup:
        epoch = 3

    for turn in range(1):
        tic = time.time()
        for i in trange(epoch):
            np.random.shuffle(train_pair)
            for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                          range(len(train_pair) // batch_size + 1)]:
                inputs = [adj_matrix, r_index, r_val, t_index, rel_matrix, ent_matrix, t_indexT]
                output = model(inputs)
                loss = align_loss(pairs, output)
                print(loss.item())

                # 反向传播和优化
                opt.zero_grad()
                loss.backward()
                opt.step()

            if i == epoch - 1:
                toc = time.time()
                model.eval()
                with torch.no_grad():
                    output = model(inputs)
                    Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1], output.cpu())
                    evaluater.test(Lvec, Rvec, 1)
                    output2 = output.cpu().numpy()
                    output2 = output2 / (np.linalg.norm(output2, axis=-1, keepdims=True) + 1e-5)
                    dto.saveobj(output2, 'embedding_of_' + save_suffix())

                    # entity_emb = entity_emb.cpu().numpy()
                    # np.save('entity_emb.npy', entity_emb)
                    #
                    # rel_emb = rel_emb.cpu().numpy()
                    # np.save('rel_emb.npy', rel_emb)
                    #
                    #
                    # time_emb = time_emb.cpu().numpy()
                    # np.save('time_emb.npy', time_emb)

                model.train()
        training_time = toc - tic
    return training_time


@torch.no_grad()
def grid_search(sim1, sim2, sim3, real_sim1, real_sim2, weight, cnt_call=1, evaluate=True):
    if evaluate:
        links = torch.LongTensor(dev_pair)
        links = torch.transpose(links, 0, 1)
        links[1] -= getLast(filename + 'ent_ids_1')
    best_weight = 0
    best_hit = 0
    best_error = 1e9
    best_error_weight = 0
    excel = []
    tic = time.time()
    for w in weight:
        if evaluate:
            weight = 0.85
            print("weight", weight)
            sim4 = weight * sim2 + (1 - weight) * sim3
            sim_new =  (sim1 + w * sim4) / (1 + w)
            sim_new = sinkhorn(sim_new)
            h1, h10, mrrt = test(sim_new, 'sinkhorn')
            if h1 > best_hit:
                best_hit = h1
                best_weight = w
                # with open("align_pair", 'w') as f:
                #     for ent in res:
                #         f.write(f"{ent}\n")
        else:
            h1, h10, mrrt = 0, 0, 0
        # real_sim_new = sinkhorn((real_sim1 + w * real_sim2) / (1 + w))

        print('Grid Search Weight=', w)
        gc.collect()
        # P = real_sim_new.cpu().numpy()
        # print('shape of P is', P.shape)
        # if np.any(np.isnan(P)):
        #     print('P has nan')
        # error, rel_err, time_err = wl_kernel.sparse_adj_sim(P)
        # if error < best_error:
        #     best_error_weight = w
        #     best_error = error
        # row = dict(weight=w, hit_1=h1, hit_10=h10, mrr=mrrt, tot_dist=error, rel_dist=rel_err, time_err=time_err)
        # excel.append(row)
    toc = time.time()
    grid_search_time = toc - tic
    df = pd.DataFrame(excel)
    df.to_excel(save_suffix(True, cnt_call=cnt_call) + '.xlsx')
    print(best_hit, best_weight)
    print(best_error, best_error_weight)
    return grid_search_time


try:
    if global_args.train_anyway:
        train()
    output = dto.readobj('embedding_of_' + save_suffix())
except:
    print('Load embedding fail, begin to train')
    training_time = train()
    output = dto.readobj('embedding_of_' + save_suffix())

output = tf.convert_to_tensor(output)
sim = cal_sims(dev_pair, output)
if global_args.sep_eval:
    sim2 = cal_sims(dev_pair2, output)
real_sim = cal_sims(dev_pair, output, right=getLast(filename + 'ent_ids_1'))

all_triples, node_size, _ = utils2.load_triples(filename, True)
sparse_rel_matrix = seu.construct_sparse_rel_matrix(all_triples, node_size)

time_suffix = str(which_file)
if global_args.time_no_agg:
    time_suffix += '_time_no_agg'
if global_args.sep_eval:
    time_suffix += '_sensitive_pair'

# time_feature = seu.get_feature(filename)
fn = 'time_sim' + time_suffix
fn2 = 'time_sim_day' + time_suffix

# if not unsup and train_ratio == 0.2:
#     fn += '_lessSeed'
# elif unsup:
#     fn += '_unsup'
try:
    time_sim = dto.readobj(fn)
    time_sim_day = dto.readobj(fn2)
    if global_args.sep_eval:
        time_sim2 = dto.readobj(fn + '2')
    print('load time sim completed')
except:
    tic = time.time()
    time_sim = multiple_sparse_ind(3, time_feature, dev_pair, sparse_rel_matrix)
    time_sim_day = multiple_sparse_ind(3, time_feature_2, dev_pair, sparse_rel_matrix)

    toc = time.time()
    time_encode_time = toc - tic
    dto.saveobj(time_sim, fn)
    dto.saveobj(time_sim_day, fn2)

    # if global_args.sep_eval:
    #     time_sim2 = multiple_sparse_ind(2, time_feature, dev_pair2, sparse_rel_matrix)
    #     dto.saveobj(time_sim2, fn + '2')
    print('finish get time sim')


# try:
#     real_time_sim = dto.readobj('real_time_sim' + time_suffix)
#     print('load real time sim completed')
# except:
#     real_time_sim = multiple_sparse_ind(2, time_feature, dev_pair, sparse_rel_matrix,
#                                         right=getLast(filename + 'ent_ids_1'))
#     dto.saveobj(real_time_sim, 'real_time_sim' + time_suffix)
#     print('finish creating Real time sim')

def sub_exps():
    print('############### TIME only Sim is ###################')
    test(sinkhorn(time_sim))
    print('############### REL only Sim is ###################')
    test(sinkhorn(sim))
    print('############### TIME Sim + REL Sim is ###################')
    test(sinkhorn((time_sim + sim) / 2))
    print('############### NO Sinkhorn is ###################')
    test((time_sim + sim) / 2)


grid_search_range = np.linspace(0.0, 2, 9)
real_time_sim = 0


def main_exp():
    grid_search_time = grid_search(sim, time_sim, time_sim_day, 0, 0, grid_search_range)
    # if global_args.sep_eval:
    #     grid_search(sim2, time_sim2, real_sim, real_time_sim, grid_search_range, cnt_call=2)
    return grid_search_time


if global_args.sub_exps:
    sub_exps()
elif global_args.time_no_agg:
    time_sim = multiple_sparse_ind(2, time_feature, dev_pair, sparse_rel_matrix,
                                   time_no_agg=True)
    real_time_sim = multiple_sparse_ind(2, time_feature, dev_pair, sparse_rel_matrix,
                                        right=getLast(filename + 'ent_ids_1'),
                                        time_no_agg=True)
    grid_search(sim, time_sim, real_sim, real_time_sim, grid_search_range)
else:
    grid_search_time = main_exp()

with open('time_of_' + save_suffix(), 'w') as f:
    f.write(f'training time : {training_time}\n')
    f.write(f'get time sim time : {time_encode_time}\n')
    f.write(f'grid search time : {grid_search_time}\n')
