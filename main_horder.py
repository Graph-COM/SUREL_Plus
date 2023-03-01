'''
Author: Haoteng Yin
Date: 2023-02-25 15:06:04
LastEditors: VeritasYin
LastEditTime: 2023-03-01 15:09:21
FilePath: /SUREL_Plus/main_horder.py

Copyright (c) 2023 by VeritasYin, All Rights Reserved. 
'''
import argparse
import time
import sys

from ogb.linkproppred import Evaluator

from sampler.random_walks import subg_matrix

from logger import Logger
from dataloader import *
from model_horder import HONet
from train import *


def main():
    parser = argparse.ArgumentParser('SUREL+ Framework for Higher-Order Pattern Prediction')

    # general model and training setting
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_ratio', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--early_stop', type=int, default=-1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--num_walks', type=int,
                        default=200, help='number of walks')
    parser.add_argument('--num_steps', type=int,
                        default=4, help='step of walks')
    parser.add_argument('--k', type=int, default=10, help='negative samples')
    parser.add_argument('--nthread', type=int, default=16,
                        help='number of threads')
    parser.add_argument('--dataset', type=str, default='tags-math', help='dataset name',
                        choices=['DBLP-coauthor', 'tags-math'])
    parser.add_argument('--metric', type=str, default='MRR', help='metric for evaluating performance',
                        choices=['AUC', 'MRR', 'Hits'])
    parser.add_argument('--use_raw', action='store_true',
                        help='whether to use raw features')
    parser.add_argument('--inf_only', action='store_true',
                        help='whether to perform inference only')
    parser.add_argument('--log_dir', type=str,
                        default='./log/', help='log directory')
    parser.add_argument('--load_model', type=str,
                        default=None, help='saved model path')
    parser.add_argument('--debug', default=False,
                        action='store_true', help='whether to use debug mode')

    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    args.metric = 'MRR'
    # setup logger and tensorboard
    rlog = Logger(args)
    logger = rlog.set_up_log(sys_argv)
    if args.nthread > 0:
        torch.set_num_threads(args.nthread)
    logger.info(f"torch num_threads {torch.get_num_threads()}")

    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    data = DE_Hyper_Dataset(args.dataset)
    G_enc = data.process(logger)

    train_hedge = (data.pos_hedge.t(), data.neg_hedge)
    val_hedge = get_pos_neg_edges(
        'valid', data.split_edge, None, data.num_nodes)
    test_hedge = get_pos_neg_edges(
        'test', data.split_edge, None, data.num_nodes)
    inf_hedge = {'train': train_hedge, 'valid': val_hedge, 'test': test_hedge}

    prep_start = time.time()
    node_idx = np.arange(G_enc.shape[0])
    x, xpe = subg_matrix(
        G_enc, node_idx, num_walks=args.num_walks, num_steps=args.num_steps)
    xpe = torch.from_numpy(xpe).to(device).float() / args.num_walks
    logger.info(f'LP Encoding Size {xpe.shape}')
    time_prep = time.time() - prep_start
    logger.info(f"Prep. Runtime: {time_prep:.2f}s")

    # define model
    model = HONet(num_layers=args.num_layers, input_dim=args.num_steps, hidden_dim=args.hidden_channels, out_dim=1,
                  x_dim=data.num_feature, dropout=args.dropout)
    model.to(device)

    evaluator = Evaluator(name='ogbl-citation2')

    if args.inf_only and args.load_model:
        load_checkpoint(f'{rlog.save_path}/{args.load_model}', model)
        sta = time.time()
        results, d_inf = eval_model_horder(
            model, x, inf_hedge, evaluator, args.batch_size, device, rpe=xpe)
        f_output(results, args.metric, logger)
        logger.info(f'T_inf {time.time() - sta:.2f}, T_test {d_inf:.2f}')
        sys.exit(0)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            loss, auc = htrain(model, x, train_hedge, optimizer,
                               args.batch_size, device, args.k, rpe=xpe)
            logger.info(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, AUC: {auc:.4f}')

            if epoch % args.eval_steps == 0:
                sta = time.time()
                results, d_inf = eval_model_horder(
                    model, x, inf_hedge, evaluator, args.batch_size, device, rpe=xpe)
                if epoch % args.log_steps == 0:
                    logger.info(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
                    f_output(results, args.metric, logger)
                    logger.info(f'T_inf {time.time() - sta:.2f}, T_test {d_inf:.2f}')
                    logger.info('---')

                if rlog.add_result(run, results):
                    checkpoint = {'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'epoch': epoch}
                    save_checkpoint(checkpoint, filename=f'{rlog.save_path}/{args.stamp}_{run}')
                    break
        rlog.print_statistics(run=run, logger=logger)
    if args.runs > 1:
        rlog.print_statistics(logger=logger)


if __name__ == "__main__":
    main()
