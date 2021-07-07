import os
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import numpy as np
import tensorflow as tf
tf.compat.v1.random.set_random_seed(1234)

from tflib.gan import ConditionalLinearGenerator, LinearGenerator
from tflib.utils import load_model_from_checkpoint


####################################################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str,
                        help='Directory for saving the model checkpoints')
    parser.add_argument('--samples_path', type=str,
                        help='path for saving the generated data (default: save to model dir)')
    parser.add_argument('--num_samples', type=int, default=20000,
                        help='num of samples')
    parser.add_argument('--conditional', action='store_true', help='If passed, generate samples with conditional input')
    parser.add_argument('--digit', type=int, default=1,
                        help='Digit to generate. Only used when --conditional is set to True. If digit=-1, generate all digits and store in .npz')
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
        OUTPUT_SIZE = 512
        Z_DIM = 128

    ### set up session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:

        ### define the varialbe for generating samples
        noise = tf.random.normal(shape=(BS, Z_DIM), dtype=tf.float32)
        if args.conditional:
            if args.digit>=0 and args.digit<=9:
                fake_labels = tf.ones(shape=(BS), dtype=tf.int32)*args.digit
                samples = ConditionalLinearGenerator(BS, fake_labels, OUTPUT_SIZE, embedding_dim=100, noise=noise)
            elif args.digit==-1:
                num = int(BS/10)
                labels_list=[]
                for digit in range(10):
                    aux_labels = np.ones(shape=(num), dtype=np.int32)*digit
                    labels_list.append(aux_labels)
                fake_labels = tf.convert_to_tensor(np.concatenate(labels_list),dtype=tf.int32)
                samples = ConditionalLinearGenerator(BS, fake_labels, OUTPUT_SIZE, embedding_dim=100, noise=noise)
            else:
                raise('Error. Please, introduce a digit from 0 to 9 or -1.')
        else:
            samples = LinearGenerator(BS, OUTPUT_SIZE, noise=noise)

        ### load the model
        vars = [v for v in tf.compat.v1.global_variables()]
        saver = tf.compat.v1.train.Saver(vars)
        sess.run(tf.compat.v1.variables_initializer(vars))
        if_load = load_model_from_checkpoint(model_dir, saver, sess)
        assert if_load is True

        ### get samples
        noise_sample = []
        img_sample = []
        label_sample = []
        for i in range(int(np.ceil(num_samples / BS))):
            if args.conditional:
                noise_batch, img_batch, label_batch = sess.run([noise, samples, fake_labels])
                label_sample.append(label_batch)
            else:
                noise_batch, img_batch = sess.run([noise, samples])
            noise_sample.append(noise_batch)
            img_sample.append(img_batch)
            
            
    noise_sample = np.concatenate(noise_sample)[:num_samples]
    img_sample = np.concatenate(img_sample)[:num_samples]
    label_sample = np.concatenate(label_sample)[:num_samples]
    
    if args.conditional and args.digit==-1:
        np.savez_compressed(os.path.join(save_dir, 'generated_images_labels.npz'), labels=label_sample, images=img_sample)
    elif args.conditional:
        np.savez_compressed(os.path.join(save_dir, 'generated_images_labels{}.npz'.format(args.digit)), labels=label_sample, images=img_sample)
    else:
        np.savez_compressed(os.path.join(save_dir, 'generated.npz'), noise=noise_sample, images=img_sample)