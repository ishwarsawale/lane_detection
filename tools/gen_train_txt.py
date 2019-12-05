import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np

def gen_train_sample(src_dir, b_gt_image_dir, image_dir):
    """
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param i_gt_image_dir:
    :param image_dir:
    :return:
    """

    with open('{:s}/train1.txt'.format(src_dir), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None:
                print('Issue {:s}'.format(image_name))
                continue
            else:
                info = '{:s} {:s}'.format(image_path, binary_gt_image_path)
                file.write(info + '\n')
    return



def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='The origin path of unzipped tusimple dataset')

    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()

    gt_image_dir = ops.join(args.src_dir, 'gt_image')
    gt_binary_dir = ops.join(args.src_dir, 'gt_binary_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)

    gen_train_sample(args.src_dir, gt_binary_dir, gt_image_dir)
