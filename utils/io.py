import logging
import os

import torch

import utils


def save_results(dir, steps, subfolder='results'):
    expr_dir = os.path.join(dir, subfolder)
    utils.mkdir(expr_dir)
    save_data_path = os.path.join(expr_dir, 'results.pth')

    steps = [(d.detach().cpu(), l.detach().cpu(), lr) for (d, l, lr) in steps]
    torch.save(steps, save_data_path)
    logging.info('Results saved to {}'.format(save_data_path))
