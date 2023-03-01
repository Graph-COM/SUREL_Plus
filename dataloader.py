from ogb.linkproppred import PygLinkPropPredDataset
from scipy.sparse import csr_matrix
from torch_sparse import coalesce
from utils import *
from torch_geometric.utils import degree, k_hop_subgraph


class LinkPropDataset():
    def __init__(self, dataset, mask_ratio=0.05, k=10, use_weight=False, use_coalesce=False, use_feature=False,
                 use_val=False):
        self.dataset = dataset
        self.data = PygLinkPropPredDataset(name=dataset)
        self.graph = self.data[0]
        self.split_edge = self.data.get_edge_split()
        self.mask_ratio = mask_ratio
        self.k = k
        self.use_feature = use_feature
        self.use_weight = (use_weight and 'edge_weight' in self.graph)
        self.use_coalesce = use_coalesce
        self.use_val = use_val
        self.gtype = 'homo' if 'mag' not in dataset else 'hetero'

        if ('vessel' in dataset) and use_feature:
            self.graph['x'] = torch.nn.functional.normalize(self.graph['x'], dim=0)

        if 'x' in self.graph:
            self.num_nodes, self.num_feature = self.graph['x'].shape
        else:
            self.num_nodes = len(torch.unique(self.graph['edge_index']))
            self.num_feature = 0

        if 'source_node' in self.split_edge['train']:
            self.directed = True
            self.train_edge = self.graph['edge_index'].t()
        else:
            self.directed = False
            self.train_edge = self.split_edge['train']['edge']

        if use_weight:
            self.train_weight = self.split_edge['train']['weight']
            if use_coalesce:
                train_edge_col, self.train_weight = coalesce(self.train_edge.t(), self.train_weight, self.num_nodes,
                                                             self.num_nodes)
                self.train_edge = train_edge_col.t()
            self.train_wmax = max(self.train_weight)
        else:
            self.train_weight = None
        # must put after coalesce
        self.len_train = self.train_edge.shape[0]

    def process(self, logger):
        if logger is not None:
            logger.info(f'{self.data.meta_info}\nKeys: {self.graph.keys}')
            logger.info(
                f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.len_train} with mask ratio {self.mask_ratio}')
            logger.info(
                f'use_weight {self.use_weight}, use_coalesce {self.use_coalesce}, use_feature {self.use_feature}, use_val {self.use_val}')

        if 'vessel' in self.dataset:
            force_undirected = True
            deg = degree(self.train_edge.t()[0])
            val, indices = torch.sort(deg)
            target = indices[val > 0]
            idx = np.random.permutation(len(target))
            idx = target[idx[:int(self.len_train * self.mask_ratio)]]
            _, edge_index, _, edge_mask = k_hop_subgraph(idx, 3, self.train_edge.t())
            self.pos_edge, obsrv_edge = edge_index.t(), self.train_edge[~edge_mask]
            self.num_pos = self.pos_edge.shape[0]
        else:
            force_undirected = False
            self.num_pos = int(self.len_train * self.mask_ratio)
            idx = np.random.permutation(self.len_train)
            # pos sample edges masked for training, observed edges for structural features
            self.pos_edge, obsrv_edge = self.train_edge[idx[:self.num_pos]], self.train_edge[idx[self.num_pos:]]

        new_edge_index, _ = add_self_loops(self.graph.edge_index)
        neg_edge = negative_sampling(new_edge_index, num_nodes=self.graph.num_nodes,
                                     num_neg_samples=self.len_train+1, force_undirected=force_undirected)
        self.neg_edge = neg_edge[:, idx[:min(self.num_pos * self.k, self.len_train)]].t()

        val_edge = self.train_edge
        self.val_nodes = torch.unique(self.train_edge).tolist()

        if self.use_weight:
            obsrv_e_weight = self.train_weight[idx[self.num_pos:]]
            val_e_weight = self.train_weight
        else:
            obsrv_e_weight = np.ones(self.len_train - self.num_pos, dtype=int)
            val_e_weight = np.ones(self.len_train, dtype=int)

        if self.use_val:
            # collab allows using valid edges for training
            obsrv_edge = torch.cat(
                [obsrv_edge, self.split_edge['valid']['edge']])
            inf_edge = torch.cat(
                [self.train_edge, self.split_edge['valid']['edge']])
            self.test_nodes = torch.unique(inf_edge).tolist()
            if self.use_weight:
                obsrv_e_weight = torch.cat(
                    [self.train_weight[idx[self.num_pos:]], self.split_edge['valid']['weight']])
                inf_e_weight = torch.cat(
                    [self.train_weight, self.split_edge['valid']['weight']], dim=0)
                if self.use_coalesce:
                    obsrv_edge_col, obsrv_e_weight = coalesce(obsrv_edge.t(), obsrv_e_weight, self.num_nodes,
                                                              self.num_nodes)
                    obsrv_edge = obsrv_edge_col.t()
                    inf_edge_col, inf_e_weight = coalesce(inf_edge.t(), inf_e_weight, self.num_nodes,
                                                          self.num_nodes)
                    inf_edge = inf_edge_col.t()
                self.inf_wmax = max(inf_e_weight)
            else:
                obsrv_e_weight = np.ones(obsrv_edge.shape[0], dtype=int)
                inf_e_weight = np.ones(inf_edge.shape[0], dtype=int)
        else:
            inf_edge, inf_e_weight = self.train_edge, self.train_weight
            self.test_nodes = self.val_nodes

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_obsrv = net_obsrv + net_obsrv.T
        assert sum(G_obsrv.diagonal()) == 0

        max_val_idx = torch.max(val_edge).item()
        net_val = csr_matrix((val_e_weight, (val_edge[:, 0].numpy(), val_edge[:, 1].numpy())),
                             shape=(max_val_idx + 1, max_val_idx + 1))
        G_val = net_val + net_val.T
        assert sum(G_val.diagonal()) == 0

        if self.use_val:
            max_full_idx = torch.max(inf_edge).item()
            net_full = csr_matrix((inf_e_weight, (inf_edge[:, 0].numpy(), inf_edge[:, 1].numpy())),
                                  shape=(max_full_idx + 1, max_full_idx + 1))
            G_full = net_full + net_full.transpose()
            assert sum(G_full.diagonal()) == 0
        else:
            G_full = G_val

        if logger is not None:
            # sparsity of graph
            logger.info(
                f'Sparsity of loaded graph {G_obsrv.getnnz() / (max_obsrv_idx + 1) ** 2}')
            # statistic of graph
            logger.info(
                f'Observed subgraph with {np.sum(G_obsrv.getnnz(axis=1) > 0)} nodes and {int(G_obsrv.nnz / 2)} edges;')
            logger.info(
                f'Training subgraph with {len(torch.unique(self.pos_edge))} nodes and {self.pos_edge.size(0)} edges.')

        # self.data = None
        print('Dataset Ready.')
        return {'train': G_obsrv, 'val': G_val, 'test': G_full}


class DEH_Dataset():
    def __init__(self, dataset, relation, mask_ratio=0.05, k=10):
        self.data = torch.load(f'./dataset/sgrl/{dataset}_{relation}.pl')
        self.split_edge = self.data['split_edge']
        self.node_type = list(self.data['num_nodes_dict'])
        self.mask_ratio = mask_ratio
        self.k = k
        rel_key = ('author', 'writes', 'paper') if relation == 'cite' else (
            'paper', 'cites', 'paper')
        self.obsrv_edge = self.data['edge_index'][rel_key]
        self.split_edge = self.data['split_edge']
        self.gtype = 'Heterogeneous' if relation == 'write' else 'Homogeneous'

        if 'x' in self.data:
            self.num_nodes, self.num_feature = self.data['x'].shape
        else:
            self.num_nodes, self.num_feature = self.obsrv_edge.unique().size(0), None

        if 'source_node' in self.split_edge['train']:
            self.directed = True
            self.train_edge = self.graph['edge_index'].t()
        else:
            self.directed = False
            self.train_edge = self.split_edge['train']['edge']

        self.len_train = self.train_edge.shape[0]

    def process(self, logger):
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.len_train} with mask ratio {self.mask_ratio}')

        self.num_pos = int(self.len_train * self.mask_ratio)
        idx = np.random.permutation(self.len_train)
        # pos sample edges masked for training, observed edges for structural features
        self.pos_edge, obsrv_edge = self.train_edge[idx[:self.num_pos]], torch.cat(
            [self.train_edge[idx[self.num_pos:]], self.obsrv_edge])

        new_edge_index, _ = add_self_loops(
            self.data['split_edge']['train']['edge'].t())
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=self.num_nodes, num_neg_samples=self.len_train)
        self.neg_edge = neg_edge[:, idx[:min(
            self.num_pos * self.k, self.len_train)]].t()

        val_edge = torch.cat([self.train_edge, self.obsrv_edge])
        len_redge = len(self.obsrv_edge)

        pos_e_weight = np.ones(self.num_pos, dtype=int)
        obsrv_e_weight = np.ones(
            self.len_train - self.num_pos + len_redge, dtype=int)
        val_e_weight = np.ones(self.len_train + len_redge, dtype=int)

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_obsrv = net_obsrv + net_obsrv.T
        assert sum(G_obsrv.diagonal()) == 0

        # subgraph for training(5 % edges, pos edges)
        max_pos_idx = torch.max(self.pos_edge).item()
        net_pos = csr_matrix((pos_e_weight, (self.pos_edge[:, 0].numpy(), self.pos_edge[:, 1].numpy())),
                             shape=(max_pos_idx + 1, max_pos_idx + 1))
        G_pos = net_pos + net_pos.T
        assert sum(G_pos.diagonal()) == 0

        max_val_idx = torch.max(val_edge).item()
        net_val = csr_matrix((val_e_weight, (val_edge[:, 0].numpy(), val_edge[:, 1].numpy())),
                             shape=(max_val_idx + 1, max_val_idx + 1))
        G_val = net_val + net_val.T
        assert sum(G_val.diagonal()) == 0

        G_full = G_val
        # sparsity of graph
        logger.info(
            f'Sparsity of loaded graph {G_obsrv.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(
            f'Observed subgraph with {np.sum(G_obsrv.getnnz(axis=1) > 0)} nodes and {int(G_obsrv.nnz / 2)} edges;')
        logger.info(
            f'Training subgraph with {np.sum(G_pos.getnnz(axis=1) > 0)} nodes and {int(G_pos.nnz / 2)} edges.')

        print('Dataset Ready.')
        return {'train': G_obsrv, 'val': G_val, 'test': G_full}


class DE_Hyper_Dataset():
    def __init__(self, dataset, mask_ratio=0.6, k=10):
        self.data = torch.load(f'./dataset/sgrl/{dataset}.pl')
        self.obsrv_edge = torch.from_numpy(self.data['edge_index'])
        self.split_edge = self.data['triplets']
        self.mask_ratio = mask_ratio
        self.k = k
        self.gtype = 'Hypergraph'

        if 'x' in self.data:
            self.num_nodes, self.num_feature = self.data['x'].shape
        else:
            self.num_nodes, self.num_feature = self.obsrv_edge.unique().size(0), None

    def get_edge_split(self, ratio, k=1000, seed=2021):
        np.random.seed(seed)
        tuples = torch.from_numpy(self.data['triplets'])
        idx = np.random.permutation(len(self.data['tuples']))
        num_train = int(ratio * self.num_tup)
        split_idx = {'train': {'hedge': tuples[idx[:num_train]]}}
        val_idx, test_idx = np.split(idx[num_train:], 2)
        split_idx['valid'], split_idx['test'] = {
            'hedge': tuples[val_idx]}, {'hedge': tuples[test_idx]}
        node_neg = torch.randint(torch.max(tuples), (len(val_idx), k))
        split_idx['valid']['hedge_neg'] = torch.cat(
            [split_idx['valid']['hedge'][:, :2].repeat(1, k).view(-1, 2).t(), node_neg.view(1, -1)]).t()
        split_idx['test']['hedge_neg'] = torch.cat(
            [split_idx['test']['hedge'][:, :2].repeat(1, k).view(-1, 2).t(), node_neg.view(1, -1)]).t()
        return split_idx

    def process(self, logger):
        self.pos_hedge = self.split_edge['train']['hedge']
        node_neg = torch.randint(
            self.num_nodes, (self.pos_hedge.size(0), self.k))
        self.neg_hedge = torch.cat([self.pos_hedge[:, :2].repeat(
            1, self.k).view(-1, 2).t(), node_neg.view(1, -1)])
        logger.info(
            f'node size {self.num_nodes}, feature dim {self.num_feature}, edge size {self.obsrv_edge.shape[0]} with mask ratio {self.mask_ratio}')
        obsrv_edge = self.obsrv_edge

        # load observed graph and save as a CSR sparse matrix
        max_obsrv_idx = torch.max(obsrv_edge).item()
        obsrv_e_weight = np.ones(len(obsrv_edge), dtype=int)
        net_obsrv = csr_matrix((obsrv_e_weight, (obsrv_edge[:, 0].numpy(), obsrv_edge[:, 1].numpy())),
                               shape=(max_obsrv_idx + 1, max_obsrv_idx + 1))
        G_enc = net_obsrv + net_obsrv.T
        assert sum(G_enc.diagonal()) == 0

        # sparsity of graph
        logger.info(
            f'Sparsity of loaded graph {G_enc.getnnz() / (max_obsrv_idx + 1) ** 2}')
        # statistic of graph
        logger.info(
            f'Observed subgraph with {np.sum(G_enc.getnnz(axis=1) > 0)} nodes and {int(G_enc.nnz / 2)} edges;')

        return G_enc
