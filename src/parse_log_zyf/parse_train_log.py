# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:30:09 2018

@author: zhaoy
"""
import os
import sys
import os.path as osp
import argparse
import re


rlt_train_detail_basename = 'train_acc_detail.txt'
rlt_train_basename = 'train_acc.txt'
rlt_verif_basename = 'verif_acc.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='Parse and Plot train log')
    # general
    parser.add_argument('log_path', default='./train-log.txt',
                        help='path to log')
    parser.add_argument('--save-dir', default='',
                        help='where to save parse results')
    parser.add_argument('--save-train-detail', action='store_true',
                        help='where to save parse results')

    args = parser.parse_args()
    return args


def parse_train_log(log_fn, save_dir=None, save_train_detail=False):
    if save_dir is None:
        save_dir = './rlt_parse_log'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp = open(log_fn, 'r')

    batches_per_epoch = 0

    fp_out_train_detail = None
    if save_train_detail:
        fn_out_train_detail = osp.join(save_dir, rlt_train_detail_basename)
        fp_out_train_detail = open(fn_out_train_detail, 'w')
        write_string = 'epoch\tbatch\tspeed(samples/sec)\tacc\n'
        fp_out_train_detail.write(write_string)

    fn_out_train = osp.join(save_dir, rlt_train_basename)
    fn_out_verif = osp.join(save_dir, rlt_verif_basename)

    fp_out_train = open(fn_out_train, 'w')
    fp_out_verif = open(fn_out_verif, 'w')

    write_string = 'epoch\ttrain-acc\ttime-cost\n'
    fp_out_train.write(write_string)

    write_string = 'epoch\tbatch\tacc_avg_verifDB'
    write_string += '\tinfer_time(lfw)\txnorm(lfw)\tacc(lfw)\tacc_std(lfw)'
    write_string += '\tinfer_time(cfp_ff)\txnorm(cfp_ff)\tacc(cfp_ff)\tacc_std(cfp_ff)'
    write_string += '\tinfer_time(cfp_fp)\txnorm(cfp_fp)\tacc(cfp_fp)\tacc_std(cfp_fp)'
    write_string += '\tinfer_time(age)\txnorm(age)\tacc(age)\tacc_std(age)\tAccuracy-Highest\n'
    fp_out_verif.write(write_string)

    train_write_string = ''
    verif_write_string = ''

    line_cnt = 0
    train_epoch_idx = 0
    test_batch_idx = 0

    verif_acc_avg = 0.0
    verif_db_cnt = 0

    for line in fp:
        line_cnt += 1

        if line.startswith('INFO:root:Epoch'):
            if 'Batch' in line:
                spl = re.split('[^0-9.]+', line)
                write_string = '\t{:10}\t{:10}\t{:10}\t{:10}\n'.format(
                    spl[1], spl[2], spl[3], spl[4]
                )

                if spl[1]=='0' and int(spl[2]) > batches_per_epoch:
                    batches_per_epoch = int(spl[2])

            if save_train_detail:
                fp_out_train_detail.write(write_string)

            if 'Train-acc' in line:
                line = line.strip()

                train_epoch_idx = re.split('\D+', line)[1]
                train_write_string += '{:10}\t{:10}'.format(
                    train_epoch_idx, line.split('=')[-1])

            if 'Time cost' in line:
                line = line.strip()

                train_write_string += '\t{:10}\n'.format(line.split('=')[-1])
                fp_out_train.write(train_write_string)
                train_write_string = ''

        if line.startswith('infer time'):
            verif_write_string += '\t{:10}'.format(line.split()[-1])

        if 'XNorm:' in line:
            verif_write_string += '\t{:10}'.format(line.split()[-1])
            if line.startswith('[lfw]'):
                test_batch_idx = re.split('\D+', line)[1]

        if 'Accuracy-Flip:' in line:
            _acc = line.split()[-1]
            acc = _acc.split('+-')

            verif_acc_avg += float(acc[0])
            verif_db_cnt += 1

            verif_write_string += '\t{:10}\t{:10}'.format(acc[0], acc[1])

        if 'Accuracy-Highest:' in line:
            verif_write_string += '\t{:10}\n'.format(line.split()[-1])
            fp_out_verif.write('{:5.2f}\t{:10}\t{:10}'.format(
                float(test_batch_idx)/batches_per_epoch,  test_batch_idx, verif_acc_avg / verif_db_cnt) + verif_write_string)

            verif_write_string = ''
            verif_acc_avg = 0.0
            verif_db_cnt = 0

        if line_cnt % 1000 == 0:
            print('---> {} lines processed'.format(line_cnt))
            fp_out_train.flush()
            fp_out_verif.flush()

            if save_train_detail:
                fp_out_train_detail.flush()

    fp_out_train.close()
    fp_out_verif.close()
    if save_train_detail:
        fp_out_train_detail.close()

    print('===> batches_per_epoch: ', batches_per_epoch)
    return batches_per_epoch


if __name__ == '__main__':

#    args = parse_args()
#    print('input args:', args)

#    log_fn = args.log_path
#    save_train_detail = args.save_train_detail

    log_fn = './train-log-r100-0221.txt'
    save_train_detail = False

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = './rlt_parse_log-' + osp.splitext(osp.basename(log_fn))[0]

    parse_train_log(log_fn, save_dir, save_train_detail)
