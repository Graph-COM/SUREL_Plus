import argparse
from ogb.linkproppred import Evaluator
from scipy.sparse import save_npz, load_npz
from random_walks import rw_matrix
import time, sys
import pprgo

from logger import Logger
from dataloader import *
from model import *
from train import *


def main():
    parser = argparse.ArgumentParser(description='SUREL+ Framework for Link / Relation Type Prediction)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_ratio', type=float, default=0.05)
    parser.add_argument('--valid_perc', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5, help='teleport probability in PPR')
    parser.add_argument('--eps', type=float, default=0.0001, help='precision of PPR approx')
    parser.add_argument('--topk', type=int, default=100, help='sample size of  node set')
    parser.add_argument('--num_walks', type=int, default=200, help='number of walks')
    parser.add_argument('--num_steps', type=int, default=4, help='step of walks')
    parser.add_argument('--k', type=int, default=10, help='negative samples')
    parser.add_argument('--nthread', type=int, default=16, help='number of threads')
    parser.add_argument('--dataset', type=str, default='ogbl-citation2', help='dataset name',
                        choices=['ogbl-ppa', 'ogbl-ddi', 'ogbl-citation2', 'ogbl-collab', 'ogbl-vessel', 'mag'])
    parser.add_argument('--relation', type=str, default='cite', help='relation type',
                        choices=['write', 'cite'])
    parser.add_argument('--metric', type=str, default='MRR', help='metric for evaluating performance',
                        choices=['AUC', 'MRR', 'Hits'])
    parser.add_argument('--aggrs', type=str, default='mean', choices=['mean', 'lstm', 'attn'],
                        help='type of set neural encoder')
    parser.add_argument('--sencoder', type=str, default='LP', choices=['LP', 'ppr', 'spd', 'deg'],
                        help='type of structure encoder')
    parser.add_argument('--reduced', action='store_true', help='whether to compress structural features')
    parser.add_argument('--use_raw', action='store_true', help='whether to use raw features as input')
    parser.add_argument('--use_node_embedding', action='store_true', help='whether to load node embedding')
    parser.add_argument('--use_weight', action='store_true', help='whether to use edge weight as input')
    parser.add_argument('--use_val', action='store_true', help='whether to use val as input')
    parser.add_argument('--load_ppr', action='store_true', help='whether to load precomputed ppr')
    parser.add_argument('--save_ppr', action='store_true', help='whether to save calculated ppr')
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')
    parser.add_argument('--debug', default=False, action='store_true', help='whether to use debug mode')

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    alpha = args.alpha
    topk = args.topk
    eps = args.eps
    args.reduced = True

    # customized for each dataset
    if 'ddi' in args.dataset:
        args.metric = 'Hits@20'
    elif 'collab' in args.dataset:
        args.metric = 'Hits@50'
        args.use_val = True
        alpha = 0.7
    elif 'ppa' in args.dataset:
        args.metric = 'Hits@100'
        alpha = 0.5
    elif 'citation' in args.dataset:
        args.metric = 'MRR'
        alpha = 0.1
    elif 'vessel' in args.dataset:
        args.use_raw = True
        args.metric = 'AUC'
    elif 'mag' in args.dataset:
        args.metrc = 'MRR'
    else:
        raise NotImplementedError

    # setup logger and tensorboard

    if 'MRR' in args.metric:
        rlog = Logger(args.runs, args)
    elif 'Hits' in args.metric:
        rlogs = {
            'Hits@10': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
        rlog = rlogs[args.metric]
    elif 'AUC' in args.metric:
        rlog = Logger(args.runs, args)
    else:
        raise NotImplementedError

    logger = rlog.set_up_log(args, sys_argv)
    if args.nthread > 0:
        torch.set_num_threads(args.nthread)
    logger.info(f"torch num_threads {torch.get_num_threads()}")

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if 'mag' in args.dataset:
        data = DEH_Dataset(args.dataset, args.relation)
        args.x_dim = len(data.node_type)
    else:
        data = LinkPropDataset(args.dataset, args.train_ratio, args.k,
                               use_weight=args.use_weight,
                               use_coalesce=args.use_weight,
                               use_feature=args.use_raw,
                               use_val=args.use_val)
    graphs = data.process(logger)

    train_edge = (data.pos_edge.t(), data.neg_edge.t())
    if 'mag' in args.dataset:
        val_edge = get_pos_neg_edges('valid', data.split_edge, data.split_edge['train']['edge'], data.num_nodes)
        test_edge = get_pos_neg_edges('test', data.split_edge, data.split_edge['train']['edge'], data.num_nodes)
    else:
        val_edge = get_pos_neg_edges('valid', data.split_edge, data.graph.edge_index, data.graph.num_nodes,
                                     percent=args.valid_perc)
        test_edge = get_pos_neg_edges('test', data.split_edge, data.graph.edge_index, data.num_nodes)
    inf_edge = {'train': train_edge, 'valid': val_edge, 'test': test_edge}

    if args.use_raw:
        embed = data.graph.x
        if args.use_node_embedding:
            embedding = torch.load('embedding.pt', map_location='cpu')
            embed = torch.cat([embed, embedding], dim=-1)
        embed = embed.to(device).float()
    else:
        embed = None

    G_obsrv, G_inf = graphs['train'], graphs['test']

    prep_start = time.time()
    train_idx = np.arange(G_obsrv.shape[0])
    inf_idx = np.arange(G_inf.shape[0])
    if args.sencoder == 'LP':
        # obtain node sets and LP for training
        x, xpe = rw_matrix(G_obsrv, train_idx, num_walks=args.num_walks, num_steps=args.num_steps, reduced=args.reduced)
        # obtain node sets and LP for inference
        z, zpe = rw_matrix(G_inf, inf_idx, num_walks=args.num_walks, num_steps=args.num_steps, reduced=args.reduced)
        if args.reduced:
            xpe = torch.from_numpy(xpe).to(device).float() / args.num_walks
            zpe = torch.from_numpy(zpe).to(device).float() / args.num_walks
            logger.info(f'Encoding Size {xpe.shape}, {zpe.shape}')
    else:
        x = pprgo.topk_ppr_matrix(G_obsrv, alpha, eps, train_idx, topk, normalization='sym')
        x, xpe = encoding(x, G_obsrv, args.sencoder)
        if args.load_ppr:
            z_path = f'{args.dataset}_z_{alpha}_{topk}_{eps}.npz'
            try:
                z = load_npz(z_path)
            except FileNotFoundError:
                logger.info(f'{z_path} does not exist.')
                sys.exit(0)
        else:
            # compute the ppr vectors for train/val nodes using ACL's ApproximatePR
            z = pprgo.topk_ppr_matrix(G_inf, alpha, eps, inf_idx, topk, normalization='sym')
            z, zpe = encoding(z, G_inf, args.sencoder)
        args.num_steps = 1
    time_prep = time.time() - prep_start
    logger.info(f"Encoding {args.sencoder} Runtime: {time_prep:.2f}s")
    del graphs

    if args.save_ppr:
        save_npz(f'{args.dataset}_z_{alpha}_{topk}_{eps}', z)

    predictor = Net(num_layers=args.num_layers, input_dim=args.num_steps, hidden_dim=args.hidden_channels, out_dim=1,
                    x_dim=data.num_feature, use_feature=args.use_raw, dropout=args.dropout, aggrs=args.aggrs).to(device)
    evaluator = Evaluator(name=args.dataset) if 'mag' not in args.dataset else Evaluator(name='ogbl-citation2')

    train_edges = torch.cat(train_edge, dim=1)
    y = torch.cat([torch.ones(train_edge[0].size(1)), torch.zeros(train_edge[1].size(1))]).to(device)

    # LSTM-based encoder needs the size of various squeezes for aggregation
    ptr = True if args.aggrs != 'lstm' else False

    for run in range(args.runs):
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            loss, auc = train(predictor, x, train_edges, y, optimizer, args.batch_size, device, args.k, ptr=ptr,
                              feature=embed, rpe=xpe)
            logger.info(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

            if epoch % args.eval_steps == 0:
                sta = time.time()
                if 'MRR' in args.metric:
                    results = inference_mrr(predictor, x, z, inf_edge, evaluator, args.batch_size, device, ptr=ptr,
                                            rpe=[xpe, zpe])
                else:
                    results = inference(predictor, x, z, inf_edge, evaluator, args.batch_size, device,
                                        ptr=ptr, feature=embed, rpe=[xpe, zpe], metric=args.metric)
                dt = time.time() - sta

                if epoch % args.log_steps == 0:
                    if 'Hits' in args.metric:
                        for key, result in results.items():
                            train_hits, valid_hits, test_hits = result
                            logger.info(key)
                            logger.info(f'Run: {run + 1:02d}, '
                                        f'Epoch: {epoch:02d}, '
                                        f'Loss: {loss:.4f}, '
                                        f'Valid: {100 * valid_hits:.2f}%, '
                                        f'Test: {100 * test_hits:.2f}%')
                    else:
                        train_mrr, valid_mrr, test_mrr = results
                        logger.info(f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Valid: {valid_mrr:.4f}, '
                                    f'Test: {test_mrr:.4f}')
                    logger.info(f'T_inf {dt:.2f}')
                    logger.info('---')

                if 'Hits' in args.metric:
                    for key, result in results.items():
                        if rlogs[key].add_result(run, result) and (key == args.metric):
                            break
                else:
                    if rlog.add_result(run, results):
                        break

        if args.metric in ['MRR', 'AUC']:
            rlog.print_statistics(run=run, logger=logger)
        else:
            for key in rlogs.keys():
                logger.info(key)
                rlogs[key].print_statistics(run=run, logger=logger)

    if args.metric in ['MRR', 'AUC']:
        rlog.print_statistics(logger=logger)
    else:
        for key in rlogs.keys():
            logger.info(key)
            rlogs[key].print_statistics(logger=logger)


if __name__ == "__main__":
    main()
