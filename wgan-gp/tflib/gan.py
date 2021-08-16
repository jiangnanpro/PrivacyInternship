import time
import os

import numpy as np
import tensorflow as tf
tf.compat.v1.random.set_random_seed(1234)

from tflib.ops.linear import Linear
from tflib.ops.conv2d import Conv2D
from tflib.ops.batchnorm import Batchnorm
from tflib.ops.deconv2d import Deconv2D
from tflib.ops.embedding import Embedding
from tflib.plot import plot, flush, tick
from tflib.save_images import save_images
from tflib.utils import shuffle_in_unison
import tflib as lib


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, DIM=64, OUTPUT_DIM=28*28, MODE='wgan-gp', noise=None):
    if noise is None:
        noise = tf.random.normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('Generator.BN3', [0,2,3], output)

    output = Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, INPUT_WIDTH=28, INPUT_HEIGHT=28, DIM=64, MODE='wgan-gp'):
    output = tf.reshape(inputs, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])

    output = Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def ConditionalGenerator(n_samples, labels, embedding_dim=100, DIM=64, OUTPUT_DIM=28*28, MODE='wgan-gp', noise=None):
    assert labels.shape[0]==n_samples
    if noise is None:
        noise = tf.random.normal([n_samples, 128])
    else:
        assert labels.shape[0]==noise.shape[0]
    label_embedding = Embedding('ConditionalGenerator.Embedding', 11, embedding_dim, labels)
    
    noise_labels = tf.concat([noise, label_embedding],1)

    #embeddings = tf.keras.layers.Embedding(10, 10)
    #embed = embeddings(words_ids)

    output = Linear('ConditionalGenerator.Input', 128+embedding_dim, 4*4*4*DIM, noise_labels)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalGenerator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = Deconv2D('ConditionalGenerator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalGenerator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = Deconv2D('ConditionalGenerator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalGenerator.BN3', [0,2,3], output)

    output = Deconv2D('ConditionalGenerator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ConditionalDiscriminator(inputs, labels, embedding_dim=100, INPUT_WIDTH=28, INPUT_HEIGHT=28, DIM=64, MODE='wgan-gp'):
    assert labels.shape[0]==inputs.shape[0]
    labels_in = Embedding('ConditionalDiscriminator.Embedding', 11, embedding_dim, labels)
    labels_in = Linear('ConditionalDiscriminator.Labels', embedding_dim, INPUT_WIDTH*INPUT_HEIGHT, labels_in)
    labels_in = tf.reshape(labels_in, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])

    images_in = tf.reshape(inputs, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])

    output = tf.concat([images_in, labels_in], axis=1)


    output = Conv2D('ConditionalDiscriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = Conv2D('ConditionalDiscriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalDiscriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = Conv2D('ConditionalDiscriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = Batchnorm('ConditionalDiscriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = Linear('ConditionalDiscriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def ResidualLayer(name, n_in, n_out, inputs):
    output = Linear(name+'.Linear', n_in, n_out, inputs)
    output = Batchnorm(name+'.BN', [0,1], output)
    output = tf.nn.relu(output)
    return tf.concat([output,inputs], axis=1)


def TabularGenerator(n_samples, n_features, DIM=(256,256), noise=None):
    z_dim = 128
    if noise is None:
        noise = tf.random.normal([n_samples, z_dim])
    input_dim = z_dim
    output = noise
    for i,elem in enumerate(DIM):
        output = ResidualLayer('TabularResidualLayer{}'.format(i+1), input_dim, elem, output)
        input_dim+=elem
    output = Linear('TabularGenerator.Output', input_dim, n_features, output)
    output = tf.nn.relu(output)
    return output

def TabularDiscriminator(inputs, DIM=(256,256)):
    n_features = int(inputs.shape[1])
    input_dim = n_features
    output = inputs
    for i,elem in enumerate(DIM):
        output = Linear('TabularLinearDiscriminator.{}'.format(i+1),input_dim, elem, output)        
        output = tf.nn.leaky_relu(output)
        output = tf.nn.dropout(output, rate=0.4)
        input_dim = elem
    output = Linear('TabularDiscriminator.Output', input_dim, 1, output)
    return output

def ConditionalLinearGenerator(n_samples, labels, n_features, embedding_dim=100, noise=None):
    assert labels.shape[0]==n_samples
    if noise is None:
        noise = tf.random.normal([n_samples, 128])
    else:
        assert labels.shape[0]==noise.shape[0]
    label_embedding = Embedding('ConditionalLinearGenerator.Embedding', 11, embedding_dim, labels)
    
    noise_labels = tf.concat([noise, label_embedding],1)

    output = Linear('ConditionalLinearGenerator.Input', 128+embedding_dim, n_features*2, noise_labels)
    output = tf.nn.relu(output)
    output = Linear('ConditionalLinearGenerator.2', n_features*2, round(1.5*n_features), output)
    output = tf.nn.relu(output)
    output = Linear('ConditionalLinearGenerator.Output', round(1.5*n_features), n_features, output)
    output = tf.nn.relu(output)

    return output

def ConditionalLinearDiscriminator(inputs, labels, embedding_dim=100, DIM=64):
    assert labels.shape[0]==inputs.shape[0]
    n_features = int(inputs.shape[1])
    labels_in = Embedding('ConditionalLinearDiscriminator.Embedding', 11, embedding_dim, labels)
    # TO DO: remove INPUT_WIDTH and INPUT_HEIGHT

    output = tf.concat([inputs, labels_in], axis=1)

    output = Linear('ConditionalLinearDiscriminator.1', n_features+embedding_dim, DIM, output)
    output = tf.nn.leaky_relu(output)
    output = Linear('ConditionalLinearDiscriminator.2', DIM, DIM*2, output)
    output = tf.nn.leaky_relu(output)
    output = Linear('ConditionalLinearDiscriminator.3', DIM*2, DIM*4, output)
    output = tf.nn.leaky_relu(output)
    output = Linear('ConditionalLinearDiscriminator.Output', DIM*4, 1, output)

    return output

def train(IMAGES, LABELS, INPUT_WIDTH, INPUT_HEIGHT, MODEL_PATH, BATCH_SIZE=50, DIM=64, MODE='wgan-gp', LAMBDA=10, CRITIC_ITERS=5, ITERS=100000,
          TRAIN_WITH_DP=False, L2_NORM_CLIP=None, NOISE_MULTIPLIER=None, OUTPUT_IMAGES_PATH=None):

    OUTPUT_DIM = INPUT_WIDTH*INPUT_HEIGHT
    if OUTPUT_IMAGES_PATH is None:
        OUTPUT_IMAGES_PATH = MODEL_PATH

    real_data = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    fake_data = Generator(BATCH_SIZE, DIM, OUTPUT_DIM, MODE)

    disc_real = Discriminator(real_data, INPUT_WIDTH, INPUT_HEIGHT, DIM, MODE)
    disc_fake = Discriminator(fake_data, INPUT_WIDTH, INPUT_HEIGHT, DIM, MODE)
    
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')


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
        gen_cost = -tf.compat.v1.reduce_mean(disc_fake)
        disc_cost = tf.compat.v1.reduce_mean(disc_fake) - tf.compat.v1.reduce_mean(disc_real)

        alpha = tf.compat.v1.random.uniform(
            shape=[BATCH_SIZE,1], 
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.compat.v1.gradients(Discriminator(interpolates), [interpolates])[0]
        slopes = tf.compat.v1.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.compat.v1.reduce_mean((slopes-1.)**2)
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
                learning_rate=1e-4, 
                beta1=0.5,
                beta2=0.9
            )
        gen_train_op = gen_train_op.minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params)

        clip_disc_weights = None

    elif MODE == 'dcgan':
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.ones_like(disc_fake)
        ))

        disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_fake, 
            labels=tf.zeros_like(disc_fake)
        ))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_real, 
            labels=tf.ones_like(disc_real)
        ))
        disc_cost /= 2.

        gen_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5
        ).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=2e-4, 
            beta1=0.5
        ).minimize(disc_cost, var_list=disc_params)

        clip_disc_weights = None

    # For saving samples
    fixed_noise = tf.compat.v1.constant(np.random.normal(size=(128, 128)).astype('float32'))
    fixed_noise_samples = Generator(128, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        save_images(
            samples.reshape((128, INPUT_WIDTH, INPUT_HEIGHT)), 
            os.path.join(OUTPUT_IMAGES_PATH,'samples_{}_{}.png'.format(frame, MODE))
        )

    dev_ratio = 0.1
    dev_size = 0
    while dev_size<(IMAGES.shape[0]*dev_ratio):
        dev_size = dev_size+BATCH_SIZE
    # Dataset iterator
    train_size = IMAGES.shape[0]-dev_size
    train_images = IMAGES[:train_size]
    train_labels = LABELS[:train_size]
    dev_images = IMAGES[train_size:]
    dev_labels = LABELS[train_size:]
    
    train_gen = data_generator((train_images, train_labels),BATCH_SIZE)
    dev_gen = data_generator((dev_images, dev_labels),BATCH_SIZE)
    def inf_train_gen():
        while True:
            for images,targets in train_gen():
                yield images

    # Train loop
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as session:

        session.run(tf.compat.v1.global_variables_initializer())

        gen = inf_train_gen()

        for iteration in range(ITERS):
            start_time = time.time()

            if iteration > 0:
                _ = session.run(gen_train_op)

            if MODE == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = CRITIC_ITERS
            for _ in range(disc_iters):
                _data = next(gen)
                _disc_cost, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_data: _data}
                )
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)

            plot('train disc cost', _disc_cost)
            plot('time', time.time() - start_time)

            # Calculate dev loss, save weights and generate samples every 10000 iters
            if iteration % 10000 == 9999:
                if MODEL_PATH:
                    saver.save(session, os.path.join(MODEL_PATH,'{}'.format(MODE)))
                dev_disc_costs = []
                for images,_ in dev_gen():
                    _dev_disc_cost = session.run(
                        disc_cost, 
                        feed_dict={real_data: images}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                plot('dev disc cost', np.mean(dev_disc_costs))                

                generate_image(iteration, _data)

            # Write logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                flush(MODEL_PATH)

            tick()

def data_generator(data, batch_size):
    images, targets = data

    shuffle_in_unison(images,targets)
    def get_epoch():
        shuffle_in_unison(images,targets)
        image_batches = images.reshape(-1, batch_size, int(images.shape[1]*images.shape[2]))
        target_batches = targets.reshape(-1, batch_size)
        for i in range(len(image_batches)):
            yield (np.copy(image_batches[i]), np.copy(target_batches[i]))

    return get_epoch