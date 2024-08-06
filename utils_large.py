import io
import json
from typing import *

import pickle
import torch
import torch.nn.utils.rnn as rnn
import torch_sparse
import numpy as np
from torch import Tensor
from torch_scatter import scatter, scatter_max, scatter_min
from torch_geometric.utils import softmax
import torch
from torch import Tensor
from functools import partial
import time


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


def sim(x1: Tensor, x2: Tensor):
    return torch.mm(x1, x2.transpose(0, 1))


def l1_dist(x1: Tensor, x2: Tensor):
    return torch.cdist(x1, x2, p=1.0)


def l2_dist(x1: Tensor, x2: Tensor):
    return torch.cdist(x1, x2, p=2.0)


def cosine_distance(x1, x2, eps=1e-8):
    return 1 - cosine_sim(x1, x2, eps)


def to_tensor(device, dtype, *args):
    return apply(lambda x: torch.tensor(x, device=device, dtype=dtype), *args)


def orthogonal_projection(W: torch.Tensor) -> Tensor:
    try:
        u, s, v = torch.svd(W)
    except:  # torch.svd may have convergence issues for GPU and CPU.
        try:
            u, s, v = torch.svd(W + 1e-4 * W.mean() * torch.rand_like(W))
        except:
            return W
    return torch.mm(u, v.t())


def has_key(mp, k):
    return mp is not None and k in mp and mp[k] is not None


def apply(func, *args):
    if func is None:
        func = lambda x: x
    lst = []
    for arg in args:
        lst.append(func(arg))
    return tuple(lst)


def norm_process(embed: torch.Tensor, eps=1e-5) -> torch.Tensor:
    n = embed.norm(dim=1, p=2, keepdim=True)
    embed = embed / (n + eps)
    return embed


def norm_embed(embed: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return norm_process(embed)


def dict_values_to_tensor(d: {}, device="cuda"):
    dict_tensor = [torch.tensor(d[i], dtype=torch.long, device=device) for i in range(len(d))]
    packed = rnn.pack_sequence(dict_tensor, False)
    return packed


def seperate_index_type(graph):
    return graph["edge_index"], graph["edge_type"]


def lst_argmax(lst: List[Any], min=False):
    func = torch.argmin if min else torch.argmax
    original_size = lst[0].size()
    t = []
    for i in lst:
        t.append(i.view(1, -1))

    t = torch.cat(t, dim=0)
    t = func(t, dim=0)
    return t.view(original_size)


def print_size(*args, **kwargs):
    print("---PRINTSIZE---")
    for k, v in kwargs.items():
        if v is None:
            print(k, "is None")
        elif isinstance(v, Tensor):
            print(k, v.size())
        else:
            print(k, v)

    print("---PRINT END---")


def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in range(n):
        fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def apply_on_sparse(func, tensor):
    tensor = tensor.coalesce()
    values = func(tensor._values())
    return ind2sparse(tensor._indices(), tensor.size(), values=values)


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


# def ind2sparseNd(ind:Tensor, sizes)

def rebuild_with_indices(sp: Tensor):
    sp = sp.coalesce()
    return ind2sparse(sp._indices(), sp.size(0), sp.size(1)).coalesce()


def rdpm(total, cnt):
    return torch.randperm(total)[:cnt]


def procrustes(emb1, emb2, link0, link1):
    u, s, v = torch.svd(emb2[link1].t().mm(emb1[link0]))
    return u.mm(v.t())


def z_score(embed):
    mean = torch.mean(embed, dim=0)
    std = torch.std(embed, dim=0)
    return (embed - mean) / std


def get_iv(sps: List[Tensor]):
    iv = []
    for sp in sps:
        a = sp.coalesce()
        iv += [a._indices(), a._values()]

    return tuple(iv)


def sparse_softmax(x: Tensor, dim=0):
    v = softmax(x._values(), x._indices()[dim])
    return ind2sparse(x._indices(), x.size(), values=v)


def batch_spspmm(a, b, batch_size=1000, verbose=10, filter_softmax=0.01):
    n = a.size(0)
    m = a.size(1)
    assert m == b.size(0)
    k = b.size(1)
    ai, av, bi, bv = get_iv([a, b])
    alen = av.numel()
    result = None
    print('batch spmm begins')
    for id_begin in range(0, alen, batch_size):
        id_end = min(alen, id_begin + batch_size)
        i, v = torch_sparse.spspmm(ai[:, id_begin:id_end], av[id_begin:id_end], bi, bv, n, m, k)
        curr = torch.sparse_coo_tensor(i, v, [n, k])
        result = curr if result is None else result + curr
        # result = result.coalesce()
        result = filter_which(sparse_softmax(result), eps=filter_softmax)
        if id_begin % batch_size * verbose == 0:
            print('batch spmm, total', alen, 'complete', id_begin,
                  ' total result=', result._values().size())

    return result


def spspmm(a, b, separate=False):
    n = a.size(0)
    m = a.size(1)
    assert m == b.size(0)
    k = b.size(1)
    ai, av, bi, bv = get_iv([a, b])
    # print('n,m,k=', n, m, k)
    # print('ai, av, bi, bv=', apply(lambda x: x.numel(), ai, av, bi, bv))
    i, v = torch_sparse.spspmm(ai, av, bi, bv, n, m, k)
    if separate:
        nonzero_mask = v != 0.
        return i[:, nonzero_mask], v[nonzero_mask], [n, k]

    return torch.sparse_coo_tensor(i, v, [n, k])


def spmm_ds(d: Tensor, s: Tensor) -> Tensor:
    return spmm(s.t(), d.t()).t()


def spmm_sd(s: Tensor, d: Tensor) -> Tensor:
    s = s.coalesce()
    i, v, s, t = s._indices(), s._values(), s.size(0), s.size(1)
    return torch_sparse.spmm(i, v, s, t, d)


def spmm(s: Tensor, d: Tensor) -> Tensor:
    if s.is_sparse and d.is_sparse:
        return spspmm(s, d)
    elif s.is_sparse:
        return spmm_sd(s, d)
    elif d.is_sparse:
        return spmm_ds(s, d)
    else:
        return s.mm(d)


def masked_minmax(a: Tensor, eps=1e-8, masked_val=0., in_place=True):
    mask = a != masked_val
    if mask.sum().item() == 0:
        return a
    aa = a
    a = minmax(a[mask], eps=eps)
    if not in_place:
        aa = aa.clone().detach()
    aa[mask] = a
    return aa


def minmax(a: Tensor, dim=-1, eps=1e-8, in_place=True) -> Tensor:
    if a.is_sparse:
        return sparse_minmax(a, eps, in_place)
    a = a - a.min(dim, keepdim=True)[0]
    a = a / (eps + a.max(dim, keepdim=True)[0])
    return a


def sparse_minmax(a: Tensor, eps=1e-8, in_place=True) -> Tensor:
    a_x = a._values()
    a_x = minmax(a_x, eps=eps)
    if in_place:
        a._values().copy_(a_x)
        return a

    ret = ind2sparse(a._indices(), a.size(), values=a_x)
    return ret


def to_torch_sparse(matrix, dtype=float, device="cuda"):
    matrix = matrix.tocoo()
    return ind2sparse(torch.LongTensor([matrix.row.tolist(), matrix.col.tolist()]),
                      values=torch.tensor(matrix.data.astype(dtype)), size=tuple(matrix.shape)).to(device)


@torch.no_grad()
def dense_to_sparse_mini_batch(X, batch_size=5000):
    inds, vals = [], []
    for begin in range(0, X.size(0), batch_size):
        end = min(X.size(0), begin + batch_size)
        ind, val = dense_to_sparse(X[begin:end], True)
        ind[0] += begin
        inds.append(ind)
        vals.append(val)

    inds = torch.cat(inds, dim=-1)
    vals = torch.cat(vals, dim=0)
    x_typename = torch.typename(X).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    return sparse_tensortype(inds, vals, X.size()).coalesce()


def dense_to_sparse(x, sep=False):
    if x.is_sparse:
        return x
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    if sep:
        return indices, values
    return sparse_tensortype(indices, values, x.size()).coalesce()


def scatter_op(tensor: Tensor, op="sum", dim=-1, dim_size=None):
    tensor = tensor.coalesce()
    return scatter(tensor._values(), tensor._indices()[dim], reduce=op, dim_size=dim_size)


def sparse_max(tensor: Tensor, dim=-1):
    tensor = tensor.coalesce()
    return scatter_max(tensor._values(), tensor._indices()[dim], dim_size=tensor.size(dim))


def sparse_min(tensor: Tensor, dim=-1):
    tensor = tensor.coalesce()
    return scatter_min(tensor._values(), tensor._indices()[dim], dim_size=tensor.size(dim))


def sparse_argmax(tensor, scatter_dim, dim=0):
    tensor = tensor.coalesce()
    argmax = sparse_max(tensor, dim)[1]
    argmax[argmax == tensor._indices().size(1)] = 0
    return tensor._indices()[scatter_dim][argmax]


def sparse_argmin(tensor, scatter_dim, dim=0):
    tensor = tensor.coalesce()
    return tensor._indices()[scatter_dim][sparse_min(tensor, dim)[1]]


def mp2list(mp, assoc=None):
    if assoc is None:
        return sorted(list(mp.keys()), key=lambda x: mp[x])
    if isinstance(assoc, Tensor):
        assoc = assoc.cpu().numpy()
    return mp2list({k: assoc[v] for k, v in mp.items()}, None)


def random_split(y: Tensor, total=15000, cnt_test=9000, cnt_train=4500, dim=1, device='cuda'):
    sel = torch.randperm(total, device=device)
    sel_test = sel[:cnt_test]
    sel_train = sel[cnt_test: cnt_test + cnt_train]
    sel_val = sel[cnt_test + cnt_train:]
    test = y.index_select(dim, sel_test)
    train = y.index_select(dim, sel_train)
    val = y.index_select(dim, sel_val)
    return train, test, val


def sparse_dense_element_wise_op(sparse: Tensor, dense: Tensor, op=torch.mul):
    sparse = sparse.coalesce()
    assert sparse.dim() == 2
    ind, val = sparse._indices(), sparse._values()
    val = op(val, dense[ind[0], ind[1]])
    return ind2sparse(ind, sparse.size(), values=val)


def matrix_argmax(tensor: Tensor, dim=1):
    assert tensor.dim() == 2
    if tensor.is_sparse:
        return sparse_argmax(tensor, dim, 1 - dim)
    else:
        return torch.argmax(tensor, dim)


def matrix_argmin(tensor: Tensor, dim=1):
    assert tensor.dim() == 2
    if tensor.is_sparse:
        return sparse_argmin(tensor, dim, 1 - dim)
    else:
        return torch.argmin(tensor, dim)


def topk2spmat(val0, ind0, size, dim=0, device: torch.device = 'cuda', split=False):
    if isinstance(val0, np.ndarray):
        val0, ind0 = torch.from_numpy(val0), \
                     torch.from_numpy(ind0)

    val0 = val0.to(device)
    ind0 = ind0.to(device)
    if split:
        return val0, ind0, size

    ind_x = torch.arange(size[dim]).to(device)
    ind_x = ind_x.view(-1, 1).expand_as(ind0).reshape(-1)
    ind_y = ind0.reshape(-1)
    ind = torch.stack([ind_x, ind_y])
    val0 = val0.reshape(-1)
    filter_invalid = torch.logical_and(ind[0] >= 0, ind[1] >= 0)
    ind = ind[:, filter_invalid]
    val0 = val0[filter_invalid]
    return ind2sparse(ind, list(size), values=val0)


def to_dense(x):
    if isinstance(x, Tensor) and x.is_sparse:
        return x.to_dense()
    return x


def remain_topk_sim(matrix: Tensor, dim=0, k=1500, split=False):
    # print(matrix.size())
    if matrix.is_sparse:
        matrix = matrix.to_dense()
    val0, ind0 = torch.topk(matrix, dim=1 - dim, k=min(k, int(matrix.size(1 - dim))))
    return topk2spmat(val0, ind0, matrix.size(), dim, matrix.device, split)


@torch.no_grad()
def save_similarity_matrix(sparse=False, **kwargs):
    save = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        # save[k] = v.clone().detach().cpu()
        if sparse and not v.is_sparse:
            save[k] = remain_topk_sim(v).clone().detach().cpu()
        else:
            save[k] = v.clone().detach().cpu()
    return save
    # torch.save(save, path)
    #     v = v.to_dense()
    # np.save(path + k, v.cpu().numpy())


@torch.no_grad()
def resize_sparse(x: Tensor, new_size, ind_shift):
    xi, xv = x._indices(), x._values()
    for i, shift in enumerate(ind_shift):
        if shift == 0:
            continue
        xi[i] += shift
    return ind2sparse(xi, new_size, values=xv)


@torch.no_grad()
def filter_which(x: Tensor, **kwargs):
    # ind_0 : gt 0
    # val: lt 0.01
    # ind_1  : eq 2
    # ind_1: eq [2,4,56]
    if not x.is_sparse:
        filter_which(dense_to_sparse(x), **kwargs)

    def cmpn(op, tensor, val):
        if not isinstance(val, Iterable):
            return op(tensor, val)
        mask = None
        multiple_op = isinstance(op, Iterable)
        for i, item in enumerate(val):
            curr = op[i](tensor, item) if multiple_op else op(tensor, item)
            mask = curr if mask is None else torch.logical_and(mask, curr)
        return mask

    ind, val = x._indices(), x._values()
    mask = None
    for k, v in kwargs.items():
        cmd = k.split('_')
        op, v = v
        if cmd[0] == 'ind':
            dim = int(cmd[1])
            curr = cmpn(op, ind[dim], v)
        elif cmd[0] == 'val':
            curr = cmpn(op, val, v)
        else:
            continue
        mask = curr if mask is None else torch.logical_and(mask, curr)
    mask = torch.arange(val.numel())[mask].to(ind.device)
    # print('total', mask.numel(), 'total remain', mask.sum())
    return ind2sparse(ind.index_select(1, mask), x.size(), values=val[mask])


def split_sp(sp: Tensor):
    return sp._indices(), sp._values(), sp.size()
