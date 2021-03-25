import os, sys
sys.path.append(os.getcwd())

import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.nist
import tflib.plot

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])


def train():
    real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    fake_data = Generator(BATCH_SIZE)

    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')


    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        gen_train_op = tf.train.RMSPropOptimizer(
            learning_rate=5e-5
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(
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

        alpha = tf.random_uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

        if TRAIN_WITH_DP:
            gen_train_op = DPAdamGaussianOptimizer(
                l2_norm_clip=L2_NORM_CLIP,
                noise_multiplier=NOISE_MULTIPLIER,
                num_microbatches=1,
                learning_rate=1e-4,
                beta1=0.5,
                beta2=0.9
                )
        else:
            gen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, 
                beta1=0.5,
                beta2=0.9
            )
        gen_train_op = gen_train_op.minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params)

        clip_disc_weights = None

    elif MODE == 'dcgan':
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_fake, 
            tf.ones_like(disc_fake)
        ))

        disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_fake, 
            tf.zeros_like(disc_fake)
        ))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_real, 
            tf.ones_like(disc_real)
        ))
        disc_cost /= 2.

        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5
        ).minimize(disc_cost, var_list=disc_params)

        clip_disc_weights = None

    # For saving samples
    fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    fixed_noise_samples = Generator(128, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        lib.save_images.save_images(
            samples.reshape((128, INPUT_WIDTH, INPUT_HEIGHT)), 
            'samples_{}_{}_nist.png'.format(frame, MODE)
        )

    # Dataset iterator
    train_gen, dev_gen, test_gen = lib.nist.load(DATAPATH, BATCH_SIZE, BATCH_SIZE)
    def inf_train_gen():
        while True:
            for images,targets in train_gen():
                yield images

    # Train loop
    with tf.Session() as session:

        session.run(tf.initialize_all_variables())

        gen = inf_train_gen()

        for iteration in range(ITERS):
            start_time = time.time()

            if iteration > 0:
                _ = session.run(gen_train_op)

            if MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = CRITIC_ITERS
            for i in range(disc_iters):
                _data = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)

            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 10000 == 9999:
                dev_disc_costs = []
                for images,_ in dev_gen():
                    _dev_disc_cost = session.run(
                        disc_cost, 
                        feed_dict={real_data: images}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

                generate_image(iteration, _data)

            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()

            lib.plot.tick()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN with nist dataset after preprocessing')
    parser.add_argument('--num_iters', type=int, default=100000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=50, help='Size of the batch')
    parser.add_argument('--critic_iters', type=int, default=8, help='For WGAN and WGAN-GP, number of critic iters per gen iter')
    parser.add_argument('--dim', type=int, default=64, help='Model dimensionality')
    parser.add_argument('--lambda_val', type=int, default=10, help='Gradient penalty lambda hyperparameter')
    parser.add_argument('--mode', choices=['wgan-gp', 'wgan', 'dcgan'], help='Architecture and type of the generative model', default='wgan-gp')
    parser.add_argument('--datapath', help='Path for NIST data.', required=True)
    parser.add_argument('--dp', action='store_true', help='If passed, train with differential privacy')
    parser.add_argument('--l2_norm_clip', type=float, default=1.0, help='Value used to clip the gradients. Only when training with Differential Privacy')
    parser.add_argument('--noise_multiplier', type=float, default=1.1, help='Multiplier used to add noise to the gradients. Only when training with Differential Privacy')

    #parser.add_argument('--model_path', help='path for saving model file.', required=True)

    args = parser.parse_args()

    INPUT_HEIGHT = 28
    INPUT_WIDTH = 28
    OUTPUT_DIM = INPUT_HEIGHT*INPUT_WIDTH  # Number of pixels in NIST

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

    train()