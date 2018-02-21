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
    parser.add_argument('--log-path', default='./rlt_parse_log',
                        help='path to log')
    parser.add_argument('--save-dir', default='',
                        help='where to save parse results')
    parser.add_argument('--save-train-detail', action='store_true',
                        help='where to save parse results')
    parser.add_argument('--batchs-per-epoch', type=int, default=0,
                        help='batches in each epoch, see your train-log to get this value')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('input args:', args)

    log_fn = args.log_path
    batchs_per_epoch = args.batchs_per_epoch
    save_train_detail = args.save_train_detail

    # log_fn = './train-log-r100-0221.txt'
    # batchs_per_epoch = 7920
    # save_train_detail = False

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = './rlt_parse_log-' + osp.splitext(osp.basename(log_fn))[0]

    parse_train_log(log_fn, save_dir, save_train_detail)
    load_results_and_plot(save_dir, batchs_per_epoch)
