'''
Author: Haoteng Yin
Date: 2023-02-25 15:06:04
LastEditors: VeritasYin
LastEditTime: 2023-03-01 14:25:21
FilePath: /SUREL_Plus/sampler/random_walks.py

Copyright (c) 2023 by VeritasYin, All Rights Reserved. 
'''
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import fastremap
from subg_acc import gset_sampler, walk_sampler
import time
from utils import *
from dataloader import *
from tqdm import tqdm


def gen_batch(iterable, n=1, keep=False):
    length = len(iterable)
    if keep:
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]
    else:
        for ndx in range(0, length - n, n):
            yield iterable[ndx:min(ndx + n, length)]


def np_sampling(ptr, neighs, bsize, target, num_walks=200, num_steps=4):
    key, freq = [], []
    with tqdm(total=len(target), ncols=60) as pbar:
        for batch in gen_batch(target, bsize, True):
            _, freqs = walk_sampler(ptr, neighs, batch, num_walks=num_walks, 
                                    num_steps=num_steps, replacement=True)
            node_id, node_freq = freqs[:, 0], freqs[:, 1]
            key.append(node_id)
            freq.append(node_freq)
            pbar.update(len(batch))
    return np.concatenate(key), np.vstack(np.hstack(freq))


def construct_sparse(neighbors, weights, shape):
    # Note, here assume train_idx is from 0 to len(train_idx)
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(
        map(len, neighbors), dtype=int))
    j = np.concatenate(neighbors)
    return csr_matrix((weights, (i, j)), shape)


def rw_matrix(G, train_idx, num_walks=200, num_steps=4, batch_size=2000, reduced=True):
    gsize = G.shape[0]
    neighbors, freqs = np_sampling(G.indptr, G.indices, batch_size, train_idx, 
                                   num_walks=num_walks, num_steps=num_steps - 1)
    if reduced:
        proj = [(num_walks + 1) ** i for i in reversed(range(num_steps))]
        idy = freqs @ proj
        val, idx = fastremap.unique(idy, return_index=True)
        idy = fastremap.remap(
            idy, dict(zip(val, np.arange(len(idx)))), in_place=True)
        freqs = freqs[idx]
    else:
        idy = np.arange(len(freqs))
    z = construct_sparse(neighbors, idy + 1, shape=(gsize, gsize))
    freqs = np.insert(freqs, 0, np.zeros((1, num_steps)), axis=0)
    return z, freqs


def subg_matrix(G, train_idx, num_walks=200, num_steps=4):
    print(f'Start sampling for #{len(train_idx)} nodes with {num_walks} {num_steps}-step walks')
    gsize = G.shape[0]
    nsize, remap, enc = gset_sampler(G.indptr, G.indices, train_idx, 
                                     num_walks=num_walks, num_steps=num_steps - 1)
    z = csr_matrix((remap[1]+1, (np.repeat(train_idx, nsize), remap[0])), (gsize, gsize))
    assert (z.has_sorted_indices)
    enc = np.insert(enc, 0, np.zeros((1, num_steps)), axis=0)
    return z, enc


if __name__ == "__main__":
    dataset = 'ogbl-ppa'
    train_ratio = 0.05
    data = LinkPropDataset(dataset, train_ratio,
                           use_weight=False,
                           use_coalesce=False,
                           use_val=False)
    graphs = data.process(None)
    G_obsrv, G_inf = graphs['train'], graphs['test']

    prep_start = time.time()
    train_idx = np.arange(G_obsrv.shape[0])
    test_idx = np.arange(G_inf.shape[0])
    x, freq = rw_matrix(G_obsrv, train_idx)
    z, enc = subg_matrix(G_inf, train_idx)
    # x = sampler.pprgo.topk_ppr_matrix(G_obsrv, alpha, eps, train_idx, topk, normalization='sym')
    print(f'Prep Time: {time.time() - prep_start:.4f}s')
