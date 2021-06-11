import os
import numpy as np
import tensorflow as tf
tf.compat.v1.random.set_random_seed(1234)
import argparse
import pickle

from gan_nist import Generator
from tflib.utils import load_model_from_checkpoint, save_image_grid


####################################################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str,
                        help='Directory for saving the model checkpoints')
    parser.add_argument('--samples_path', type=str,
                        help='path for saving the generated data (default: save to model dir)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='num of samples')
    return parser.parse_args()


####################################################################################################################

if __name__ == '__main__':
    ### load config
    args = parse_args()
    num_samples = args.num_samples
    model_dir = args.model_path
    out_dir = args.samples_path
    save_dir = model_dir if out_dir is None else out_dir
    config_path = os.path.join(model_dir, 'params.pkl')
    BS = 100
    if os.path.exists(config_path):
        config = pickle.load(open(os.path.join(model_dir, 'params.pkl'), 'r'))
        OUTPUT_SIZE = config['OUTPUT_SIZE']
        GAN_TYPE = config['Architecture']
        Z_DIM = config['Z_DIM']
    else:        
        INPUT_WIDTH = 28
        INPUT_HEIGHT = 28
        OUTPUT_SIZE = INPUT_WIDTH*INPUT_HEIGHT
        Z_DIM = 128

    ### define the varialbe for generating samples
    noise = tf.random_normal(shape=(BS, Z_DIM), dtype=tf.float32)
    samples = Generator(BS, noise=noise)

    ### set up session
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf_config)
    init = tf.compat.v1.global_variables()
    sess.run(init)

    ### set up saver for loading model
    load_saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.global_variables()])
    _ = load_model_from_checkpoint(model_dir, load_saver, sess)

    ### get samples
    noise_sample = []
    img_sample = []
    for i in range(int(np.ceil(num_samples / BS))):
        noise_batch, img_batch = sess.run([noise, samples])
        noise_sample.append(noise_batch)
        img_sample.append(img_batch)
    noise_sample = np.concatenate(noise_sample)[:num_samples]
    img_sample = np.concatenate(img_sample)[:num_samples]
    img_sample = np.reshape(img_sample, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])
    save_image_grid(img_sample[:100], os.path.join(save_dir, 'samples.png'), [-1, 1], [10, 10])

    img_r01 = (img_sample + 1.) / 2.
    #img_r01 = img_r01.transpose(0, 2, 3, 1)  # NCHW => NHWC
    np.savez_compressed(os.path.join(save_dir, 'generated.npz'),
                        noise=noise_sample,
                        img_r01=img_r01)