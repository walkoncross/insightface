# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:45:34 2018

@author: zhaoy
"""

import os
import sys
import os.path as osp

from parse_train_log import parse_train_log
from plot_parse_results import load_results_and_plot


if __name__ == '__main__':
    log_fn = './train-log-r100-0221.txt'
    save_dir = './rlt_parse_log'

    batchs_per_epoch = 7920
    save_train_detail = False

    if len(sys.argv) > 1:
        log_fn = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    if len(sys.argv) > 3:
        batchs_per_epoch = int(sys.argv[2])

    parse_train_log(log_fn, save_dir, save_train_detail)
    load_results_and_plot(save_dir, batchs_per_epoch)