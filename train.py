import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score
import threading
import time

NCOL = 72


def gather(edge, x, device, ptr=True, encode=None):
    # obtain subgraph neighbors of left and right endpoints
    xl, xr = x[edge[0]], x[edge[1]]
    # convert to boolean mask
    lmask, rmask = xl > 0, xr > 0

    # keep index for set aggregation
    if ptr:
        indptr = torch.from_numpy(np.concatenate(
            [xl.indptr[:-1], xl.indptr[-1] + xr.indptr])).long().to(device)
    else:
        # obtain neighborhood size
        lsize, rsize = torch.from_numpy(lmask.getnnz(axis=1)).long(
        ), torch.from_numpy(rmask.getnnz(axis=1)).long()
        # generate index
        lind, rind = torch.arange(len(lsize), dtype=torch.long).repeat_interleave(lsize).to(device), \
            torch.arange(len(lsize), 2 * len(lsize), dtype=torch.long).repeat_interleave(rsize).to(device)
        indptr = torch.cat([lind, rind])
    # join ops, x+y*x_mask

    xrl, xlr = xr.multiply(lmask) + lmask, xl.multiply(rmask) + rmask
    if encode is not None:
        xl = np.stack([xl.data, xrl.data - 1]).T
        xr = np.stack([xr.data, xlr.data - 1]).T
        xz = encode[np.vstack([xl, xr])]
    else:
        xl = torch.from_numpy(
            np.stack([xl.data, xrl.data - 1]).T).float().to(device)
        xr = torch.from_numpy(
            np.stack([xr.data, xlr.data - 1]).T).float().to(device)
        xz = torch.cat([xl, xr]).unsqueeze(dim=-1)

    return xz, indptr


def hgather(hedge, x, device, encode=None):
    # obtain subgraph neighbors of left and right endpoints
    xu, xv, xw = x[hedge[0]], x[hedge[1]], x[hedge[2]]
    # convert to boolean mask
    umask, vmask, wmask = xu > 0, xv > 0, xw > 0
    # obtain neighborhood size
    usize, vsize, wsize = torch.from_numpy(umask.getnnz(axis=1)).long(), torch.from_numpy(
        vmask.getnnz(axis=1)).long(), torch.from_numpy(wmask.getnnz(axis=1)).long()
    # keep index for segment aggregation
    node_size = torch.cat([usize, wsize, vsize, wsize])
    ind = torch.arange(
        len(usize) * 4, dtype=torch.long).repeat_interleave(node_size).to(device)
    # join ops, x*y_mask+
    xwu, xuw = xw.multiply(umask) + umask, xu.multiply(wmask) + wmask
    xwv, xvw = xw.multiply(vmask) + vmask, xv.multiply(wmask) + wmask
    if encode is not None:
        xu = np.stack([xu.data, xwu.data - 1]).T
        xv = np.stack([xv.data, xwv.data - 1]).T
        xw0 = np.stack([xw.data, xuw.data - 1]).T
        xw1 = np.stack([xw.data, xvw.data - 1]).T
        xz = encode[np.vstack([xu, xw0, xv, xw1])]
    else:
        raise NotImplementedError
    assert xz.size(0) == ind.size(0)
    return xz, ind


def bgather(edge, x, out):
    # obtain subgraph neighbors of left and right endpoints
    xl, xr = x[edge[0]], x[edge[1]]
    # convert to boolean mask
    lmask, rmask = xl > 0, xr > 0
    xrl, xlr = xr.multiply(lmask) + lmask, xl.multiply(rmask) + rmask

    xl = np.stack([xl.data, xrl.data - 1]).T
    xr = np.stack([xr.data, xlr.data - 1]).T
    out[0], out[1] = xl, xr
    out[2], out[3] = lmask.getnnz(axis=1), rmask.getnnz(axis=1)


def pgather(edge, M, device, encode, gather_func, ptr=True, njobs=4):
    out_blocks = np.empty((njobs, 4), dtype=object)
    edge_blocks = np.array_split(edge, njobs, axis=1)

    threads = []
    for i in range(njobs):
        th = threading.Thread(target=gather_func, args=(edge_blocks[i], M, out_blocks[i]))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    if njobs > 1:
        xz = np.vstack([*out_blocks[:, 0], *out_blocks[:, 1]])
        xz = encode[xz] if encode is not None else torch.from_numpy(xz).float().to(device).unsqueeze(dim=-1)
        indptr = torch.from_numpy(np.concatenate(
            [[0], *out_blocks[:, 2], *out_blocks[:, 3]])).long()
        if ptr:
            return xz, indptr.cumsum(dim=0).to(device)
        else:
            return xz, torch.arange(len(indptr)-1, dtype=torch.long).repeat_interleave(indptr[1:]).to(device)

    return out_blocks


def train(predictor, g, edges, label, optimizer, batch_size, device, k, ptr=True, feature=None, rpe=None):
    predictor.train()

    total_loss = 0
    total_sample = 0
    labels, preds = [], []
    pbar = tqdm(DataLoader(range(edges.size(1)), batch_size, shuffle=True), ncols=NCOL)
    for perm in pbar:
        optimizer.zero_grad()
        edge = edges[:, perm]
        embed = feature[edge] if feature is not None else None
        target = label[perm]
        labels.append(target)
        x, ind = pgather(edge, g, device, rpe, bgather, ptr=ptr)
        pred = predictor.forward(x, ind, feature=embed)
        preds.append(pred.detach().sigmoid())
        loss = BCEWithLogitsLoss()(pred, target)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        total_sample += len(target)
        total_loss += loss.item() * len(target)
        pbar.set_description(f"Train Loss {loss.item():.4f}")
    predictions = torch.cat(preds).cpu()
    labels = torch.cat(labels).cpu()
    return total_loss / total_sample, roc_auc_score(labels, predictions)


def htrain(predictor, g, train_edge, optimizer, batch_size, device, k, feature=None, rpe=None):
    predictor.train()

    pos_train_edge, neg_train_edge = train_edge

    edges = torch.cat(train_edge, dim=1)
    label = torch.cat([torch.ones(pos_train_edge.size(1)),
                      torch.zeros(neg_train_edge.size(1))]).to(device)
    total_loss = 0
    total_sample = 0
    labels, preds = [], []
    pbar = tqdm(DataLoader(range(edges.size(1)), batch_size, shuffle=True), ncols=NCOL)
    for perm in pbar:
        optimizer.zero_grad()
        edge = edges[:, perm]
        embed = feature[edge] if feature is not None else None
        target = label[perm]
        labels.append(target)
        x, ind = hgather(edge, g, device, encode=rpe)
        pred = predictor(x, ind, feature=embed)
        preds.append(pred.detach().sigmoid())
        loss = BCEWithLogitsLoss()(pred, target)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        total_sample += len(target)
        total_loss += loss.item() * len(target)
        pbar.set_description(f"Train Loss {loss.item():.4f}")
    predictions = torch.cat(preds).cpu()
    labels = torch.cat(labels).cpu()
    return total_loss / total_sample, roc_auc_score(labels, predictions)


@torch.no_grad()
def inference(predictor, x, z, inf_edge, evaluator, batch_size, device, ptr=True, feature=None, rpe=None, metric='Hits'):
    predictor.eval()

    def test_split(split, embed, encode):
        print(f'Inf {split}')
        pos_edge, neg_edge = inf_edge[split]
        pos_preds = []
        for perm in tqdm(DataLoader(range(pos_edge.size(1)), batch_size), ncols=NCOL):
            edge = pos_edge[:, perm]
            x_inf, ind = gather(edge, embed, device, ptr=ptr, encode=encode)
            z = feature[edge] if feature is not None else None
            pos_preds += [predictor(x_inf, ind, feature=z).squeeze()]
        pos_pred = torch.cat(pos_preds, dim=0).sigmoid()

        if split != 'train':
            neg_preds = []
            for perm in tqdm(DataLoader(range(neg_edge.size(1)), batch_size), ncols=NCOL):
                edge = neg_edge[:, perm]
                x_inf, ind = gather(edge, embed, device,
                                    ptr=ptr, encode=encode)
                z = feature[edge] if feature is not None else None
                neg_preds += [predictor(x_inf, ind, feature=z).squeeze()]
            neg_pred = torch.cat(neg_preds, dim=0).sigmoid()
        else:
            neg_pred = None
        return pos_pred, neg_pred

    xpe, zpe = rpe
    #     pos_train_pred, _ = test_split('train', x, xpe)
    pos_valid_pred, neg_valid_pred = test_split('valid', z, zpe)
    sta = time.time()
    pos_test_pred, neg_test_pred = test_split('test', z, zpe)
    t_inf = time.time() - sta
    print(f'#valid_pos {len(pos_valid_pred)} #valid_neg {len(neg_valid_pred)} '
          f'#test_pos {len(pos_test_pred)} #test_neg {len(neg_test_pred)}')

    if 'Hits' in metric:
        results = {}
        for K in [10, 50, 100]:
            evaluator.K = K
            #             train_hits = evaluator.eval({
            #                 'y_pred_pos': pos_train_pred,
            #                 'y_pred_neg': neg_valid_pred,
            #             })[f'hits@{K}']
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (0, valid_hits, test_hits)
    elif 'AUC' in metric:
        valid_rocauc = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'rocauc']

        test_rocauc = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'rocauc']
        results = (0, valid_rocauc, test_rocauc)

    return results, t_inf


@torch.no_grad()
def inference_mrr(predictor, x, z, inf_edge, evaluator, batch_size, device, ptr=True, feature=None, rpe=None, metric=None):
    predictor.eval()

    def test_split(split, embed, encode):
        print(f'Inf {split}')
        pos_edge, neg_edge = inf_edge[split]
        assert pos_edge.size(0) < pos_edge.size(1)
        assert neg_edge.size(0) < neg_edge.size(1)
        k = neg_edge.size(1) // pos_edge.size(1)

        pos_preds = []
        for perm in tqdm(DataLoader(range(pos_edge.size(1)), batch_size), ncols=NCOL):
            edge = pos_edge[:, perm]
            x_inf, ind = pgather(edge, embed, device, encode, bgather, ptr=ptr)
            pos_preds += [predictor(x_inf, ind).squeeze()]
        pos_pred = torch.cat(pos_preds, dim=0).sigmoid()

        neg_preds = []
        for perm in tqdm(DataLoader(range(neg_edge.size(1)), batch_size), ncols=NCOL):
            edge = neg_edge[:, perm]
            x_inf, ind = pgather(edge, embed, device, encode, bgather, ptr=ptr)
            neg_preds += [predictor(x_inf, ind).squeeze()]
        neg_pred = torch.cat(neg_preds, dim=0).sigmoid().view(-1, k)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    xpe, zpe = rpe
    #     train_mrr = test_split('train', x, xpe)
    valid_mrr = test_split('valid', z, zpe)
    sta = time.time()
    test_mrr = test_split('test', z, zpe)
    return (0, valid_mrr, test_mrr), time.time() - sta


@torch.no_grad()
def eval_model_horder(predictor, x, inf_edge, evaluator, batch_size, device, feature=None, rpe=None):
    predictor.eval()

    def test_split(split, embed, encode):
        print(f'Inf {split}')
        pos_edge, neg_edge = inf_edge[split]
        assert pos_edge.size(0) < pos_edge.size(1)
        assert neg_edge.size(0) < neg_edge.size(1)
        k = neg_edge.size(1) // pos_edge.size(1)

        pos_preds = []
        for perm in tqdm(DataLoader(range(pos_edge.size(1)), batch_size), ncols=NCOL):
            edge = pos_edge[:, perm]
            x_inf, ind = hgather(edge, embed, device, encode=encode)
            pos_preds += [predictor(x_inf, ind).squeeze()]
        pos_pred = torch.cat(pos_preds, dim=0).sigmoid()

        neg_preds = []
        for perm in tqdm(DataLoader(range(neg_edge.size(1)), batch_size), ncols=NCOL):
            edge = neg_edge[:, perm]
            x_inf, ind = hgather(edge, embed, device, encode=encode)
            neg_preds += [predictor(x_inf, ind).squeeze()]
        neg_pred = torch.cat(neg_preds, dim=0).sigmoid().view(-1, k)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    #     train_mrr = test_split('train', x, rpe)
    valid_mrr = test_split('valid', x, rpe)
    sta = time.time()
    test_mrr = test_split('test', x, rpe)
    return (0, valid_mrr, test_mrr), time.time() - sta
