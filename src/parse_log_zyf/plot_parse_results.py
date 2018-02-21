# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:30:09 2018

@author: zhaoy
"""
import os
import sys
import os.path as osp

import re
import numpy as np

import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt


rlt_train_detail_basename = 'train_acc_detail.txt'
rlt_train_basename = 'train_acc.txt'
rlt_verif_basename = 'verif_acc.txt'


def load_train_results_and_plot(save_dir, train_rlt_fn=None, batchs_per_epoch=0):
    if save_dir is None:
        save_dir = './rlt_parse_log'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if train_rlt_fn is None:
        train_rlt_fn = osp.join(save_dir, rlt_train_basename)

    data = np.loadtxt(train_rlt_fn, skiprows=1)
    print('loaded data shape: ', data.shape)

    def plot_data(use_batch_idx=False):
        xx = data[:, 0]

        if not use_batch_idx:
            x_label = 'epoch'
        else:
            xx *= batchs_per_epoch
            x_label = 'batch'

        fig = plt.figure(figsize=(16, 12), dpi=100)

        plt.plot(xx, data[:, 1], 'g')

        # plt.xlim([0, 1e8])
        plt.ylim([0, 1])

        plt.xlabel(x_label)
        plt.ylabel('train acc')

        plt.grid()
        plt.legend(loc='lower right')
        plt.show()

        save_fn = 'train_acc_vs_%s_idx.png' % x_label
        save_fn = osp.join(save_dir, save_fn)

        fig.savefig(save_fn, bbox_inches='tight')

    plot_data()
    if batchs_per_epoch > 0:
        plot_data(use_batch_idx=True)


def load_verif_results_and_plot(save_dir, verif_rlt_fn=None, batchs_per_epoch=0):
    if save_dir is None:
        save_dir = './rlt_parse_log'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if verif_rlt_fn is None:
        verif_rlt_fn = osp.join(save_dir, rlt_verif_basename)

    data = np.loadtxt(verif_rlt_fn, skiprows=1)
    print('loaded data shape: ', data.shape)

    colors = ['g', 'r', 'b', 'c', 'm', 'y']
    labels = ['avg_dbs', 'lfw', 'cfp_ff', 'cfp_fp', 'age_db', 'age_db_highest']

    def plot_data(use_epoch_idx=False):
        xx = data[:, 0]
        if use_epoch_idx > 0:
            xx /= batchs_per_epoch
            x_label = 'epoch'
        else:
            x_label = 'batch'

        fig = plt.figure(figsize=(16, 12), dpi=100)

        plt.plot(xx, data[:, 1], colors[0], label=labels[0])

        for i in range(1, 5):
            plt.plot(xx, data[:, 4 * i], colors[i], label=labels[i])

#        plt.plot(xx, data[:, -1], colors[-1], label=labels[-1])

        # plt.xlim([0, 1e8])
        plt.ylim([0.5, 1])

        plt.xlabel(x_label)
        plt.ylabel('verif acc')

        plt.grid()
        plt.legend(loc='lower right')
        plt.show()

        save_fn = 'verif_acc_vs_%s_idx.png' % x_label

        save_fn = osp.join(save_dir, save_fn)
        fig.savefig(save_fn, bbox_inches='tight')

    plot_data(use_epoch_idx=False)

    if batchs_per_epoch > 0:
        plot_data(use_epoch_idx=True)


def load_results_and_plot(save_dir, batchs_per_epoch=0):
    load_verif_results_and_plot(save_dir, batchs_per_epoch=batchs_per_epoch)
    load_train_results_and_plot(save_dir, batchs_per_epoch=batchs_per_epoch)


if __name__ == '__main__':
    save_dir = './rlt_parse_log'
    batchs_per_epoch = 7920

    if len(sys.argv) > 1:
        log_fn = sys.argv[1]

    if len(sys.argv) > 2:
        batchs_per_epoch = int(sys.argv[2])

    load_verif_results_and_plot(save_dir, batchs_per_epoch=batchs_per_epoch)
    load_train_results_and_plot(save_dir, batchs_per_epoch=batchs_per_epoch)