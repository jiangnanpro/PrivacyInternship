#!/usr/bin/env python3

import os
import sys
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
from imageio import imread
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wgan-gp'))
from tflib.nist import load_nist_images

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-inpath_npy", "--images_npy_path", nargs='*', type=str, default=None,
        help='Path to training set images in .npy format')
    group.add_argument("-inpath_png", "--images_png_path", nargs='*', type=str, default=None,
        help='Path to training set images in .png format')
    parser.add_argument("-outpath", "--output_path", type=str, default=None,
        help='Path for where to store the statistics')
    parser.add_argument("-i", "--inception_path", type=str, default=None,
        help='Path to Inception model (will be downloaded if not provided)')
    args = parser.parse_args()

    #########
    # PATHS #
    #########
    if args.output_path is None:
        args.output_path = os.path.dirname(os.path.abspath(__file__)) # path for where to store the statistics

    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(args.inception_path) # download inception if necessary
    print("ok")

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    if args.images_png_path is not None:
        image_list = []
        for path in args.images_png_path:
            image_list.extend(glob.glob(os.path.join(args.data_path, '*.png')))
        images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    elif args.images_npy_path is not None:
        images=[]
        for path in args.images_npy_path:
            with open(path,'rb') as f:
                images_loaded = np.load(f)['img_r01']
                print(images_loaded[0])
                images.append(load_nist_images(images_loaded))
        images = np.vstack(images)

    print(images.shape)
    print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=256)
        np.savez_compressed(args.output_path, mu=mu, sigma=sigma)
    print("finished")