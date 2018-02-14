
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from datetime import datetime
import os.path as osp
from easydict import EasyDict as edict
import time
import json
import sys
import numpy as np
import importlib
import itertools
import argparse
import struct
import cv2
sys.path.append(osp.join(osp.dirname(__file__), '..', 'common'))
# import face_preprocess
from sklearn.preprocessing import normalize
# import facenet
# import lfw
import _init_paths
import mxnet as mx
from mxnet import ndarray as nd
# from caffe.proto import caffe_pb2

from matio import save_mat
from compare_feats import calc_similarity_cosine


if cv2.__version__.startswith('3.'):
    IMREAD_AS_GRAY = CV2.IMREAD_GRAYSCALE
    IMREAD_AS_COLOR = cv2.IMREAD_COLOR
elif:
    IMREAD_AS_GRAY = cv2.CV_LOAD_IMAGE_GRAYSCALE
    IMREAD_AS_COLOR = cv2.CV_LOAD_IMAGE_COLOR


def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    if mode == 'gray':
        img = cv2.imread(img_path, IMREAD_AS_GRAY)
    else:
        img = cv2.imread(img_path, IMREAD_AS_COLOR)
        if mode == 'rgb':
            #print('to rgb')
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))
    return img


def do_flip(data):
    for idx in xrange(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def init_input_blob(batch_size, image_shape):
    input_shape = (batch_size, image_shape[0], image_shape[1], image_shape[2])
    input_blob = np.zeros(input_shape, dtype=np.float32)

    return input_blob


def load_image_data(image_path, input_blob, idx, image_shape, use_mean=True):

    img = face_preprocess.read_image(image_path, mode='rgb')
    # print(img.shape)
    if img is None:
        print('parse image', image_path, 'error')
        return None

    if not (img.shape == (image_shape[1], image_shape[2], image_shape[0])):
        return None

    if use_mean > 0:
        v_mean = np.array([127.5, 127.5, 127.5],
                          dtype=np.float32).reshape((1, 1, 3))
        img = img.astype(np.float32) - v_mean
        img *= 0.0078125

    input_blob[idx] = np.transpose(img, (2, 0, 1))

    return True


def add_flip_to_input_blob(input_blob):
    batch_size = input_blob.shape[0] / 2
    for i in range(batch_size):
        input_blob[i + batch_size] = input_blob[i]
        do_flip(input_blob[i + batch_size])


def get_feature(input_blob, nets, flip_sim=False):
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=data)

    features = []

    for i, net in enumerate(nets):
        net.model.forward(db, is_train=False)
        outputs = net.model.get_outputs()

        for j in range(n_imgs):
            embedding = outputs[j].asnumpy().flatten()

            if add_flip:
                embedding_flip = outputs[j + n_imgs].asnumpy().flatten()
                if flip_sim:
                    sim = calc_similarity_cosine(embedding, embedding_flip)
                    print('---> Net #%d, flip_sim=%f\n' % (i, sim))
                embedding += embedding_flip

            _norm = np.linalg.norm(embedding)
            if _norm > 0:
                embedding /= _norm
            if i == 0:
                features.append(embedding)
            else:
                # features[j] += embedding
                features[j] = np.concatenate((features[j], embedding), axis=0)

    for j in range(n_imgs):
        _norm = np.linalg.norm()
        if _norm > 0:
            features[j] /= _norm

    return features, suc_flags


def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))


def main(args):
    print('===> args:\n', args)

    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fail_log_fn = osp.join(save_dir, 'failed_image_list.txt')
    fail_fp = open(fail_log_fn, 'w')

    gpuid = args.gpu
    ctx = mx.gpu(gpuid)
    nets = []

    image_shape = [int(x) for x in args.image_size.split(',')]

    batch_size = args.batch_size
    if args.add_flip:
        batch_size = int(args.batch_size / 2)

    if batch_size < 1:
        batch_size = 1

    input_batch_size = batch_size
    if args.add_flip:
        input_batch_size *= 2

    for model in args.model.split('|'):
        vec = model.split(',')
        assert len(vec) > 1
        prefix = vec[0]
        epoch = int(vec[1])
        print('loading', prefix, epoch)
        net = edict()
        net.ctx = ctx
        net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(
            prefix, epoch)
        # net.arg_params, net.aux_params = ch_dev(net.arg_params,
        # net.aux_params, net.ctx)
        all_layers = net.sym.get_internals()
        net.sym = all_layers['fc1_output']
        net.model = mx.mod.Module(
            symbol=net.sym, context=net.ctx, label_names=None)
        net.model.bind(
            data_shapes=[('data', (input_batch_size, 3, image_shape[1], image_shape[2]))])
        net.model.set_params(net.arg_params, net.aux_params)

        nets.append(net)

    img_list_fn = args.image_list
    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    img_list = []

    input_blob = init_input_blob(input_batch_size, image_shape)

    i = 0
    succ = 0

    for line in open(img_list_fn, 'r'):
        if i % 1000 == 0:
            print("===> Processed %d images, %d succeeded" % (i, succ))

        i += 1

        image_path = line.strip()
        full_path = osp.join(args.image_dir, image_path)

        ret = load_image_data(full_path, input_blob,
                              succ, image_shape, args.use_mean)
        if not ret:
            print('---> Failed to load: ', full_path)
            fail_fp.write(image_path + '\n')
            continue

        succ += 1
        img_list.append(image_path)

        if i == batch_size:
            fail_fp.flush()

            if args.add_flip:
                add_flip_to_input_blob(input_blob)

            features, suc_flags = get_feature(input_blob, nets)

            for j, fn in enumerate(img_list):
                a, b = osp.split(fn)
                sub_dir = osp.join(args.save_dir, a)
                if not osp.exists(sub_dir):
                    os.makedirs(sub_dir)

                if args.save_format is '.npy':
                    out_path = osp.join(sub_dir, b + "_feat.npy")
                    np.save(out_path, features[j])
                else:
                    out_path = osp.join(sub_dir, b + "_feat.bin")
                    # write_bin(out_path, features[j])
                    save_mat(out_path, features[j])

            img_list = []

    if len(img_list) > 1:
        print("===> Processed %d images, %d succeeded" % (i, succ))
        if args.add_flip:
            add_flip_to_input_blob(input_blob)

        features, suc_flags = get_feature(input_blob, nets)

        for j, fn in enumerate(img_list):
            a, b = osp.split(fn)
            sub_dir = osp.join(args.save_dir, a)
            if not osp.exists(sub_dir):
                os.makedirs(sub_dir)

            if args.save_format is '.npy':
                out_path = osp.join(sub_dir, b + "_feat.npy")
                np.save(out_path, features[j])
            else:
                out_path = osp.join(sub_dir, b + "_feat.bin")
                # write_bin(out_path, features[j])
                save_mat(out_path, features[j])

    fail_fp.close()


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-list', type=str, help='image list file')
    parser.add_argument('--image-dir', type=str,
                        help='image root dir if image list contains relative paths')
    parser.add_argument('--save-dir', type=str, default='./rlt-features',
                        help='where to save the features')
    parser.add_argument('--batch-size', type=int, help='', default=100)
    parser.add_argument('--image-size', type=str,
                        help='', default='3,112,112')
    parser.add_argument('--add-flip', action='store_true',
                        help='use (oringal + flip)')
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--use-mean', action='store_true')
    parser.add_argument('--save-format', type=str, default='.npy',
                        help='save output as: 1) .npy format; 2).bin: megaface format.')
    parser.add_argument('--flip-sim', action='store_true',
                        help='To test similarity between original image and its flipped version.')

    # parser.add_argument('--model', type=str, help='',
    #   default='../model/sphereface-20-p0_0_96_112_0,22|../model/sphereface-20-p0_0_96_95_0,21|../model/sphereface-20-p0_0_80_95_0,21')
    # parser.add_argument('--model', type=str, help='',
    #   default='../model/sphereface-s60-p0_0_96_112_0,31|../model/sphereface-s60-p0_0_96_95_0,21|../model/sphereface2-s60-p0_0_96_112_0,21|../model/sphereface3-s60-p0_0_96_95_0,23')
    # parser.add_argument('--model', type=str, help='',
    #   default='../model/sphereface-s60-p0_0_96_112_0,31|../model/sphereface-s60-p0_0_96_95_0,21|../model/sphereface2-s60-p0_0_96_112_0,21|../model/sphereface3-s60-p0_0_96_95_0,23|../model/sphereface-20-p0_0_96_112_0,22|../model/sphereface-20-p0_0_96_95_0,21|../model/sphereface-20-p0_0_80_95_0,21')
    # parser.add_argument('--model', type=str, help='',
    #   default='../model/spherefacei-s60-p0_0_96_112_0,135')
    # parser.add_argument('--model', type=str, help='',
    #   default='../model/spherefacei-s60-p0_0_96_95_0,95')
    parser.add_argument('--model', type=str, help='',
                        default='../model/model-r50-am-lfw/model,0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
