import torch
import numpy as np
from torch_geometric.utils import negative_sampling, add_self_loops
from sklearn.preprocessing import normalize


def encoding(x, adj, encoding='deg'):
    agg = None
    if encoding == 'deg':
        x += normalize(adj, norm='l1', axis=1)
        x_deg = x.getnnz(axis=1)
        x_deg = np.log(x_deg + 1)
        # x_deg = x_deg / x_deg.max()
        agg = x.copy()
        x.data = (x > 0).multiply(x_deg).data
    elif encoding == 'spd':
        x0 = x > 0
        x1 = adj > 0
        x2 = x1 ** 2
        x = x1 + x0.multiply(x2 * 0.5) + x0 * 0.3
        x.setdiag(2.3)
    elif encoding == 'ppr':
        x.data = (x.data + 0.1)/(x.data.max()+0.1)
    else:
        raise NotImplementedError
    return x, agg


def evaluate_hits(pos_pred, neg_pred, evaluator):
    results = {}
    for K in [10, 20, 50, 100]:
        evaluator.K = K
        res_hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = res_hits
    return results


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes, num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    elif 'hedge' in split_edge['train']:
        pos_edge = split_edge[split]['hedge']
        neg_edge = split_edge[split]['hedge_neg'].t()
        if percent < 100:
            np.random.seed(123)
            num_pos = pos_edge.size(1)
            perm = np.random.permutation(num_pos)
            perm = perm[:int(percent / 100 * num_pos)]
            pos_edge = pos_edge[:, perm]
            neg_edge = neg_edge.view(num_pos, -1, 3)[perm].reshape(-1, 3)
        pos_edge = pos_edge.t()
    else:
        raise NotImplementedError
    return pos_edge, neg_edge
