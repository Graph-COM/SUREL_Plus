import torch
import logging
import numpy as np
import os
import time
import streamtologger


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def set_up_log(self, args, sys_argv):
        log_path = os.path.join(args.log_dir, args.dataset)
        save_path = os.path.join(log_path, 'model')
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.stamp = time.strftime('%m%d%y_%H%M%S')
        file_path = os.path.join(log_path, f"{args.stamp}.log")

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info('Create log file at {}'.format(file_path))
        logger.info('Command line executed: python ' + ' '.join(sys_argv))
        logger.info('Full args parsed:')
        logger.info(args)
        if args.debug:
            streamtologger.redirect(target=logger)
        return logger

    def add_result(self, run, result, stop=3):
        assert len(result) == 3
        assert 0 <= run < len(self.results)
        self.results[run].append(result)
        r = np.array(self.results[run])[:, 1]
        if len(r) > stop:
            if len(r) - r.argmax() > stop + 1:
                return True
            if np.sort(r)[-stop] > 0.999:
                return True
        return False

    def print_statistics(self, run=None, logger=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            logger.info(f'Run {run + 1:02d}:\n'
                        f'Highest Train: {result[:, 0].max():.2f}\n'
                        f'Highest Valid: {result[:, 1].max():.2f}\n'
                        f'  Final Train: {result[argmax, 0]:.2f}\n'
                        f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            logger.info(f'All runs:')
            r1, r2, r3, r4, = best_result[:, 0], best_result[:, 1], best_result[:, 2], best_result[:, 3]
            logger.info(f'Highest Train: {r1.mean():.2f} ± {r1.std():.2f}\n'
                        f'Highest Valid: {r2.mean():.2f} ± {r2.std():.2f}\n'
                        f'  Final Train: {r3.mean():.2f} ± {r3.std():.2f}\n'
                        f'   Final Test: {r4.mean():.2f} ± {r4.std():.2f}')
