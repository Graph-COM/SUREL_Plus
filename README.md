# **SubGACC**: Subgraph Operation Accelerator
<p align="center">
    <a href="https://github.com/VeritasYin/subg_acc/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-BSD%202--Clause-red.svg"></a>
    <a href="https://github.com/VeritasYin/subg_acc/blob/master/setup.py"><img src="https://img.shields.io/badge/Version-v2.1-orange" alt="Version"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVeritasYin%2Fsubg_acc&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false"/></a>
</p>

The `SubGAcc` package is an extension library based on C and openmp to accelerate subgraph operations in subgraph-based graph representation learning (SGRL) with multithreading enabled. Follow the principles of algorithm system co-design, query-level subgraphs (of link/motif) (e.g. ego-network in canonical SGRLs) are decomposed into reusable node-level ones (e.g. walk in [SUREL](https://arxiv.org/abs/2202.13538) by `walk_sampler`, set in [SUREL+](https://github.com/VeritasYin/SUREL_Plus/blob/main/manuscript/SUREL_Plus_Full.pdf) by `gset_sampler`). Currently, `SubGAcc` consists of the following methods for the realization of scalable SGRLs:

- [New] `gset_sampler` walk-based set sampling with structure encoder of landing probability (LP) 
- `walk_sampler` walk-based subgraph sampling with relative positional encoding (RPE)
- `batch_sampler` walk-based sampling of training queries in batches
- `walk_join` online walk joining that reconstructs the query-level subgraph from node-level walk-based ones to serve queries (a set of nodes)

## Requirements
(Other versions may work, but are untested)

- python >= 3.8
- numpy >= 1.17
- gcc >= 8.4
- cmake >= 3.16
- make >= 4.2

## Installation
```
python setup.py install
```

## Functions

### gset_sampler

```
subg_acc.gset_sampler(indptr, indices, query, num_walks, num_steps, bucket=-1, nthread=-1, seed=111413) -> (numpy.array [n], numpy.array [?,2], numpy.array [?,num_steps+1])
```

Sample a node set for each node in `query` (size of `n`) through `num_walks`-many `num_steps`-step random walks on the input graph in CSR format (`indptr`, `indices`), and encodes landing probability at each step of all nodes in the sampled set as structural features of the seed node.

#### Parameters

* **indptr** *(np.array)* - CSR format index pointer array of the adjacency matrix.
* **indices** *(np.array)* - CSR format index array of the adjacency matrix.
* **query** *(np.array / list)* - Queried nodes to be sampled.
* **num_walks** *(int)* - The number of random walks.
* **num_steps** *(int)* - The number of steps in a walk.
* **bucket** *(int, optional)* - The buffer size of node set.
* **nthread** *(int, optional)* - The number of threads.
* **seed** *(int, optional)* - Random seed.

#### Returns

* **nsize** *(np.array)* - The size of each sampled node set.
* **maps** *(np.array)* - The sampled node set and the index of its associated structural features.
* **enc** *(np.array)* - The compressed (unique) encoding of landing probabilities.