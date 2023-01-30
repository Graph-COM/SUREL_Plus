from tqdm import tqdm
from dataloader import *
from utils import *
import time
from scipy.sparse import coo_matrix
from subg_acc import gset_sampler, walk_sampler
import fastremap


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
    with tqdm(total=len(target)) as pbar:
        for batch in gen_batch(target, bsize, True):
            _, freqs = walk_sampler(ptr, neighs, batch, num_walks=num_walks, num_steps=num_steps, replacement=True)
            node_id, node_freq = freqs[:, 0], freqs[:, 1]
            key.append(node_id)
            freq.append(node_freq)
            pbar.update(len(batch))
    return np.concatenate(key), np.vstack(np.hstack(freq))


def construct_sparse(neighbors, weights, shape):
    # Note, here assume train_idx is from 0 to len(train_idx)
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return coo_matrix((weights, (i, j)), shape).tocsr()


def rw_matrix(G, train_idx, num_walks=200, num_steps=4, batch_size=2000, reduced=False):
    gsize = G.shape[0]
    neighbors, freqs = np_sampling(G.indptr, G.indices, batch_size, train_idx, num_walks=num_walks,
                                   num_steps=num_steps - 1)
    if reduced:
        proj = [(num_walks + 1) ** i for i in reversed(range(num_steps))]
        idy = freqs @ proj
        val, idx = fastremap.unique(idy, return_index=True)
        idy = fastremap.remap(idy, dict(zip(val, np.arange(len(idx)))), in_place=True)
        freqs = freqs[idx]
    else:
        idy = np.arange(len(freqs))
    z = construct_sparse(neighbors, idy + 1, shape=(gsize, gsize))
    freqs = np.insert(freqs, 0, np.zeros((1, num_steps)), axis=0)
    return z, freqs


def subg_matrix(G, train_idx, num_walks=200, num_steps=4, reduced=True):
    gsize = G.shape[0]
    nsize, remap, freqs = gset_sampler(G.indptr, G.indices, train_idx, num_walks=num_walks, num_steps=num_steps - 1)
    z = csr_matrix((remap[:,1]+1, (np.repeat(train_idx, nsize), remap[:,0])), (gsize, gsize))
    assert(z.has_sorted_indices)
    freqs = np.insert(freqs, 0, np.zeros((1, num_steps)), axis=0)
    return z, freqs


if __name__ == "__main__":
    dataset = 'ogbl-ppa'
    train_ratio = 0.05
    data = LinkPropDataset(dataset, train_ratio,
                           use_weight=False,
                           use_coalesce=False,
                           use_degree=False,
                           use_val=False)
    graphs = data.process(None)
    G_obsrv, G_inf = graphs['train'], graphs['test']

    prep_start = time.time()
    train_idx = np.arange(10001)
    x, freq = rw_matrix(G_obsrv, train_idx)
    print(f'Prep Time: {time.time() - prep_start:.4f}s')
    # x = ppr.topk_ppr_matrix(G_obsrv, alpha, eps, train_idx, topk, normalization='sym')
