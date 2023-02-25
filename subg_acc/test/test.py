'''
Author: Haoteng Yin
Date: 2023-02-10 17:10:14
LastEditors: VeritasYin
LastEditTime: 2023-02-25 13:29:51
FilePath: /subg_acc/test/test.py

Copyright (c) 2023 by VeritasYin, All Rights Reserved. 
'''
import subg_acc as subg
import numpy as np
from scipy.sparse import csr_matrix


def edge2csr(file='twitter-2010.txt'):
    row, col = np.loadtxt(file, dtype=int).T
    data = np.ones(len(row), dtype=bool)
    nmax = max(row.max(), col.max())
    return csr_matrix((data, (row, col)), shape=(nmax+1, nmax+1))


G_full = edge2csr('test.edgelist')

ptr = G_full.indptr
neighs = G_full.indices

num_walks = 100
num_steps = 4
target = np.arange(G_full.shape[0])
print(subg.__file__)
nsize, remap, enc = subg.gset_sampler(
    ptr, neighs, target, num_walks=num_walks, num_steps=num_steps, nthread=16)
# check the alignment of nsize of remapping
assert nsize.sum() == remap.shape[1]
# check the boundary of node and encoding indices
assert (remap.max(axis=1) - [G_full.shape[0]-1, enc.shape[0]-1]).sum() == 0
# check the encoding of root
assert (enc[remap[1]][:, 0] == num_walks).sum() == len(target)
assert np.abs((enc[remap[1]].sum(axis=0) /
              G_full.shape[0] - num_walks).sum()) < 1e-10
nsize, remap, enc, raw_enc = subg.gset_sampler(
    ptr, neighs, target, num_walks=num_walks, num_steps=num_steps, debug=1)
assert (raw_enc[:, 0] == num_walks).sum() == len(target)
assert (enc[remap[1]] - raw_enc).sum() == 0
assert (raw_enc.max(axis=0) - num_walks).sum() == 0
print(f"Test Passed.")
