<!--
 * @Author: Haoteng Yin
 * @Date: 2023-02-24 18:29:44
 * @LastEditors: VeritasYin
 * @LastEditTime: 2023-02-25 13:29:08
 * @FilePath: /subg_acc/README.md
 * 
 * Copyright (c) 2023 by VeritasYin, All Rights Reserved. 
-->
# **SubGACC**: Subgraph Operation Accelerator
<p align="center">
    <a href="https://github.com/VeritasYin/subg_acc/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-BSD%202--Clause-red.svg"></a>
    <a href="https://github.com/VeritasYin/subg_acc/blob/master/setup.py"><img src="https://img.shields.io/badge/Version-v2.2-orange" alt="Version"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVeritasYin%2Fsubg_acc&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false"/></a>
</p>

The `SubGAcc` package is an extension library based on C and openmp to accelerate operations in subgraph-based graph representation learning (SGRL). 

Follow the principles of algorithm system co-design, queried subgraphs for target links/motifs (e.g. ego-network in canonical SGRLs) are decomposed into node-level ones (e.g. collection of walks by `walk_sampler` in [SUREL](https://arxiv.org/abs/2202.13538), set of nodes by `gset_sampler` in [SUREL+](https://github.com/VeritasYin/SUREL_Plus/blob/main/manuscript/SUREL_Plus_Full.pdf)), whose join can act as proxies of subgraphs, and can be reused among different queries.

Currently, `SubGAcc` consists of the following methods for scalable realization of SGRLs:

- `gset_sampler` node set sampling with structure encoder of landing probability (LP) 
- `walk_sampler` walk sampling with relative positional encoder (RPE)
- `batch_sampler` query sampling (a group of nodes) for mini-batch training of link prediction
- `walk_join` online joining of node-level walks to construct the proxy of subgraph for given queries (e.g. link query $Q= \lbrace u,v \rbrace$ $\to$ join sampled walks of node $u$ and $v$ as $\mathcal{G}_{Q} = \lbrace W_u \bigoplus W_v \rbrace$)

## Update
**Feb. 25, 2023**:
* Release v2.2 with more robust memory management of allocation, release and indexing (billion edges).
* Add bitwise-based hash for encoding structural features.
* Add test cases and script of wall time measure.

**Jan. 29, 2023**:
* Release v2.1 with refactored code base.
* More robust memory accessing with buffer for set sampler on large graphs (million nodes).

**Jan. 28, 2023**:
* Release v2.0 with the walk-based set sampler `gset_sampler`.

## Requirements
(Other versions may work, but are untested)

- python >= 3.8
- numpy >= 1.17
- gcc >= 8.4

## Installation
```
python setup.py install
```

## Functions

### gset_sampler

```
subg_acc.gset_sampler(indptr, indices, query, num_walks, num_steps) 
-> (numpy.array [n], numpy.array [2,?], numpy.array [?,num_steps+1])
```

Sample a node set for each node in `query` (size of `n`) through `num_walks`-many `num_steps`-step random walks on the input graph in CSR format (`indptr`, `indices`), and encodes landing probability at each step of all nodes in the sampled set as structural features of the seed node. 

For usage examples, see [test.py](https://github.com/VeritasYin/subg_acc/blob/master/test/test.py).

#### Parameters

* **indptr** *(np.array)* - Index pointer array of the adjacency matrix in CSR format.
* **indices** *(np.array)* - Index array of the adjacency matrix in CSR format.
* **query** *(np.array / list)* - Nodes are queried to be sampled.
* **num_walks** *(int)* - The number of random walks.
* **num_steps** *(int)* - The number of steps in a walk.
* **bucket** *(int, optional)* - The buffer size for sampled neighbors per node.
* **nthread** *(int, optional)* - The number of threads.
* **seed** *(int, optional)* - Random seed.

#### Returns

* **nsize** *(np.array)* - The size of sampled set for each node in `query`.
* **remap** *(np.array)* - Pairwised node id and the index of its associated structural encoding in `enc` array.
* **enc** *(np.array)* - The compressed (unique) encoding of structural features.
