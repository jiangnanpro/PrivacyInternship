import os, sys
sys.path.append(os.getcwd())
import time
import argparse
import pickle

import numpy as np
import tensorflow as tf
tf.compat.v1.random.set_random_seed(1234)

import tflib as lib
import tflib.qmnist
import tflib.plot
from tflib.gan import TabularGenerator, TabularDiscriminator

lib.print_model_settings(locals().copy())

def train():
    if PCA_NCOMPS is not None:
        OUTPUT_DIM = PCA_NCOMPS
    elif PREPROCESSING_MODEL == 'vgg19':
        OUTPUT_DIM = 512
    elif PREPROCESSING_MODEL == 'resnetV2':
        OUTPUT_DIM = 2048
    
    real_data = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

    fake_data = TabularGenerator(BATCH_SIZE, OUTPUT_DIM)

    disc_real = TabularDiscriminator(real_data, DIM)
    disc_fake = TabularDiscriminator(fake_data, DIM)

    gen_params = lib.params_with_name('TabularGenerator')
    disc_params = lib.params_with_name('TabularDiscriminator')


    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        gen_train_op = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=5e-5
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=5e-5
        ).minimize(disc_cost, var_list=disc_params)

        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var, 
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        alpha = tf.random.uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(TabularDiscriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

        if TRAIN_WITH_DP:
            from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
            gen_train_op = DPAdamGaussianOptimizer(
                l2_norm_clip=L2_NORM_CLIP,
                noise_multiplier=NOISE_MULTIPLIER,
                num_microbatches=1, # Possible problem after reducing the size of cost vector in tensorflow-privacy. Check: https://github.com/tensorflow/privacy/issues/17
                learning_rate=1e-4,
                beta1=0.5,
                beta2=0.9
                )
        else:
            gen_train_op = tf.compat.v1.train.AdamOptimizer(
                learning_rate=2e-4, 
                beta1=0.5,
                beta2=0.9
            )
        gen_train_op = gen_train_op.minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params)

        clip_disc_weights = None


    # Dataset iterator
    if PCA_NCOMPS is not None:
        pca, train_gen, dev_gen, test_gen = lib.qmnist.load_tabular(DATAPATH, BATCH_SIZE, BATCH_SIZE, PREPROCESSING_MODEL, PCA_NCOMPS)
        # Save trained PCA
        with open(os.path.join(MODEL_PATH,'PCA.pkl'),'wb') as f:
            pickle.dump(pca,f)
    else:
        train_gen, dev_gen, test_gen = lib.qmnist.load_tabular(DATAPATH, BATCH_SIZE, BATCH_SIZE, PREPROCESSING_MODEL)
    
    def inf_train_gen():
        while True:
            for images,targets in train_gen():
                targets = targets.astype('int32')
                yield images, targets

    # Train loop
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as session:

        session.run(tf.compat.v1.global_variables_initializer())

        gen = inf_train_gen()

        for iteration in range(ITERS):
            start_time = time.time()

            if iteration > 0:
                _ = session.run(gen_train_op)
            disc_iters = CRITIC_ITERS
            for _ in range(disc_iters):
                _data,_ = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)

            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)

            # Calculate dev loss, save weights and generate samples every 10000 iters
            if iteration % 10000 == 9999:
                if MODEL_PATH:
                    saver.save(session, os.path.join(MODEL_PATH,'{}_tabQMNIST'.format(MODE)))
                dev_disc_costs = []
                for dev_images,_ in dev_gen():
                    _dev_disc_cost = session.run(
                        disc_cost, 
                        feed_dict={real_data: dev_images}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))                

            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()

            lib.plot.tick()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN with QMNIST tabular dataset')
    parser.add_argument('--num_iters', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batch')
    parser.add_argument('--critic_iters', type=int, default=5, help='For WGAN and WGAN-GP, number of critic iters per gen iter')
    parser.add_argument('--dim', type=tuple, default=(256,256), help='Tuple or list of ints to set the dimension of the generator and discriminator')
    parser.add_argument('--pca_ncomps', type=int, default=None, help='Dimension size after applying PCA to the feature space. If None, do not apply PCA.')
    parser.add_argument('--lambda_val', type=int, default=10, help='Gradient penalty lambda hyperparameter')
    parser.add_argument('--mode', choices=['wgan-gp', 'wgan', 'dcgan'], help='Architecture and type of the generative model', default='wgan-gp')
    parser.add_argument('--datapath', help='Path for .pickle with QMNIST data.', required=True)
    parser.add_argument('--dp', action='store_true', help='If passed, train with differential privacy')
    parser.add_argument('--l2_norm_clip', type=float, default=1.0, help='Value used to clip the gradients. Only when training with Differential Privacy')
    parser.add_argument('--noise_multiplier', type=float, default=1.1, help='Multiplier used to add noise to the gradients. Only when training with Differential Privacy')
    parser.add_argument('--preprocessing', choices=['vgg19', 'resnetV2'], help='Pre-trained model to do the preprocessing', default='vgg19')
    parser.add_argument('--model_path', help='Path for saving model weights', default='')

    args = parser.parse_args()

    PCA_NCOMPS = args.pca_ncomps
    DATAPATH = args.datapath
    BATCH_SIZE = args.batch_size
    MODE = args.mode
    CRITIC_ITERS = args.critic_iters
    LAMBDA = args.lambda_val
    DIM = args.dim
    ITERS = args.num_iters # How many generator iterations to train for
    TRAIN_WITH_DP = args.dp
    L2_NORM_CLIP = args.l2_norm_clip
    NOISE_MULTIPLIER = args.noise_multiplier
    MODEL_PATH = args.model_path
    PREPROCESSING_MODEL = args.preprocessing

    rng = np.random.RandomState(2021)

    train()