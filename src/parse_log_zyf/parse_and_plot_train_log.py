# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:45:34 2018

@author: zhaoy
"""

import os
import sys
import os.path as osp
import argparse

from parse_train_log import parse_train_log
from plot_parse_results import load_results_and_plot


def parse_args():
    parser = argparse.ArgumentParser(description='Parse and Plot train log')
    # general
    parser.add_argument('log_path', default='./rlt_parse_log',
                        help='path to log')
    parser.add_argument('--save-dir', default='',
                        help='where to save parse results')
    parser.add_argument('--save-train-detail', action='store_true',
                        help='where to save parse results')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('input args:', args)

    log_fn = args.log_path
    save_train_detail = args.save_train_detail

    # log_fn = './train-log-r100-0221.txt'
    # save_train_detail = False

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = './rlt_parse_log-' + osp.splitext(osp.basename(log_fn))[0]

    batches_per_epoch = parse_train_log(log_fn, save_dir, save_train_detail)
    load_results_and_plot(save_dir, batches_per_epoch)
