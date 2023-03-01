import torch
import logging
import numpy as np
import os
import time
import streamtologger


class Logger(object):
    def __init__(self, args=None):
        self.info = args
        self.early_stop = args.early_stop
        self.log_path = os.path.join(args.log_dir, args.dataset)
        self.save_path = os.path.join(self.log_path, 'model')
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        args.stamp = time.strftime('%m%d%y_%H%M%S')
        self.file_path = os.path.join(self.log_path, f"{args.stamp}.log")
        if 'Hits' in args.metric:
            self.results = {
                'Hits@10': [[] for _ in range(args.runs)],
                'Hits@50': [[] for _ in range(args.runs)],
                'Hits@100': [[] for _ in range(args.runs)]
            }
        else:
            self.results = [[] for _ in range(args.runs)]

    def set_up_log(self, sys_argv):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.file_path)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info(f'Create log file at {self.file_path}')
        logger.info('Command line executed: python ' + ' '.join(sys_argv))
        logger.info('Full args parsed:')
        logger.info(self.info)
        if self.info.debug:
            streamtologger.redirect(target=logger)
        return logger

    def add_result(self, run, result):
        if isinstance(result, tuple):
            self.results[run].append(result)
            r = self.results[run]
        elif isinstance(result, dict):
            for key, val in result.items():
                self.results[key][run].append(val)
            r = self.results[self.info.metric][run]
        else:
            raise NotImplementedError

        assert len(r[-1]) == 3
        r = np.array(r)[:, 1]
        if len(r) > self.early_stop > 0:
            if len(r) - r.argmax() > self.early_stop:
                return True
            if np.sort(r)[-self.early_stop] > 0.9999:
                return True
        return False

    def print_statistics(self, run=None, logger=None, key=None):
        if isinstance(self.results, dict) and (key is None):
            for key in self.results.keys():
                self.print_statistics(run, logger, key)
        else:
            if key is None:
                output = 'self.results'
                key = self.info.metric
            else:
                output = 'self.results[key]'

            if run is not None:
                result = 100 * torch.tensor(eval(output)[run])
                argmax = result[:, 1].argmax().item()
                logger.info(f'Run {run + 1:02d} {key}:\n'
                            # f'Highest Train: {result[:, 0].max():.2f}\n'
                            f'Highest Valid: {result[:, 1].max():.2f}\n'
                            # f'  Final Train: {result[argmax, 0]:.2f}\n'
                            f'   Final Test: {result[argmax, 2]:.2f}')
            else:
                best_results = []
                for r in eval(output):
                    r = 100 * torch.tensor(r)
                    train1 = r[:, 0].max().item()
                    valid = r[:, 1].max().item()
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test = r[r[:, 1].argmax(), 2].item()
                    best_results.append((train1, valid, train2, test))

                best_result = torch.tensor(best_results)

                logger.info(f'All runs {key}:')
                r2, r4 = best_result[:, 1], best_result[:, 3]
                valid_std = r2.std() if len(r2) > 1 else 0
                test_std = r4.std() if len(r4) > 1 else 0
                logger.info(
                    # f'Highest Train: {r1.mean():.2f} ± {r1.std():.2f}\n'
                    f'Highest Valid: {r2.mean():.2f}±{valid_std:.2f}\n'
                    # f'  Final Train: {r3.mean():.2f} ± {r3.std():.2f}\n'
                    f'   Final Test: {r4.mean():.2f}±{test_std:.2f}')
