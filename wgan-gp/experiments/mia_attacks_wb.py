import os
import sys
import pickle
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))
sys.path.append(os.path.join(os.path.dirname(file_path),'attacks/tools/lpips_tensorflow'))

import numpy as np
from torchvision.datasets import QMNIST
import tensorflow as tf
if tf.__version__[0]=='2':
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()

from attacks.wb import optimize_z
from tflib.nist import load_nist_images
import lpips_tf
from tflib.gan import Generator
from tflib.utils import grey2RGB, load_model_from_checkpoint, check_folder, save_files, shuffle_in_unison

"""### Get qmnist-nist index correspondence"""

nist_datapath = '/home/rafael07/privacy_benchmark/data/nist'
qmnist_datapath = '/home/rafael07/privacy_benchmark/data/qmnist'
nist_qmnist_indexes_datapath = os.path.join(file_path,'qmnist_nist_indexes.pickle')

with open(nist_qmnist_indexes_datapath, 'rb') as f:
    qmnist_nist_dict = pickle.load(f)
qmnist_indexes = qmnist_nist_dict['qmnist_indexes']
nist_indexes = qmnist_nist_dict['nist_indexes']
# Randomly selecting the images for each partition
shuffle_in_unison(qmnist_indexes, nist_indexes)

qmnist_data = QMNIST(qmnist_datapath, download=True, what='nist')
qmnist_images = qmnist_data.data.numpy()
qmnist_labels = qmnist_data.targets.numpy()[:,0]

def load_whole_nist(datapath, hsf_list=[0,1,2,3,4,6,7]):
    images_stack = []
    labels_stack = []
    for hsf in hsf_list:
        with open(os.path.join(datapath, 'HSF_'+str(hsf)+'_images.npy'),'rb') as f:
            images = load_nist_images(np.load(f))
        with open(os.path.join(datapath,'HSF_'+str(hsf)+'_labels.npy'),'rb') as f:
            labels = np.load(f).reshape([-1,1])
        images_stack.append(np.array(images))
        labels_stack.append(np.array(labels))
    return np.vstack(images_stack), np.vstack(labels_stack)

nist_images, nist_labels = load_whole_nist(nist_datapath)



INPUT_HEIGHT = 28
INPUT_WIDTH = 28

LAMBDA2 = 0.2
LAMBDA3 = 0.001
RANDOM_SEED = 1000

Z_DIM = 128
maxfunc = 1000
initialize_type = 'zero'
distance = 'l2-lpips'
if_norm_reg = True

training_set_sizes = [128, 256, 412, 1024, 2048, 4096, 8192, 16384]

for n in training_set_sizes:
    load_dir = os.path.join(os.path.dirname(file_path),'models/wgan-gp_nist{}'.format(n))
    save_dir = os.path.join(load_dir,'mia_results/wb')
    nn_dir = os.path.join(load_dir,'mia_results/fbb')

    generate = np.load(os.path.join(load_dir, 'generated.npz'))
    gen_imgs = generate['images']
    gen_z = generate['noise']
    gen_feature = np.reshape(gen_imgs, [len(gen_imgs), -1])
    gen_feature = 2. * gen_feature - 1.
    gen_feature = (gen_feature*255).astype('uint8')

    x_defender_nist = nist_images[nist_indexes[:n]]
    x_defender_qmnist = qmnist_images[qmnist_indexes[:n]]

    x_reserve_nist = nist_images[nist_indexes[n:]]
    x_reserve_qmnist = qmnist_images[qmnist_indexes[n:]]

    dev_ratio = 0.1
    dev_size = 0
    while dev_size<(n*dev_ratio):
        dev_size = dev_size+32
    ### WHITE-BOX ATTACK
    
    ### load data
    pos_query_imgs = x_defender_nist[:n-dev_size]
    DATA_NUM = min(n,1000)
    BATCH_SIZE = min(DATA_NUM,20)
    pos_query_imgs = pos_query_imgs[:DATA_NUM]
    neg_query_imgs = x_reserve_nist[:DATA_NUM]
    if distance=='l2-lpips':
        n_channels = 3
        pos_query_imgs = np.apply_along_axis(grey2RGB,-1,pos_query_imgs).reshape(pos_query_imgs.shape[0],INPUT_WIDTH, INPUT_HEIGHT,3)
        neg_query_imgs = np.apply_along_axis(grey2RGB,-1,neg_query_imgs).reshape(neg_query_imgs.shape[0],INPUT_WIDTH, INPUT_HEIGHT,3)
    else:
        n_channels = 1
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        ### define variables
        
        x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=(None, INPUT_WIDTH, INPUT_HEIGHT, n_channels), name='x')

        ### initialization
        init_val_ph = None
        init_val = {'pos': None, 'neg': None}
        if initialize_type == 'zero':
            z = tf.compat.v1.Variable(tf.compat.v1.zeros([BATCH_SIZE, Z_DIM], tf.compat.v1.float32), name='latent_z')
        elif initialize_type == 'random':
            np.random.seed(RANDOM_SEED)
            init_val_np = np.random.normal(size=(Z_DIM,))
            init = np.tile(init_val_np, (BATCH_SIZE, 1)).astype(np.float32)
            z = tf.compat.v1.Variable(init, name='latent_z')
        elif initialize_type == 'nn':
            init_val['pos'] = np.load(os.path.join(nn_dir, 'pos_z.npy'))[:, 0, :]
            init_val['neg'] = np.load(os.path.join(nn_dir, 'neg_z.npy'))[:, 0, :]
            init_val_ph = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, name='init_ph', shape=(BATCH_SIZE, Z_DIM))
            z = tf.compat.v1.Variable(init_val_ph, name='latent_z')
        else:
            raise NotImplementedError

        ### get the reconstruction (x_hat)
        x_hat = Generator(BATCH_SIZE, noise=z)
        x_hat = tf.compat.v1.reshape(x_hat, [-1, 1, INPUT_WIDTH, INPUT_HEIGHT])
        x_hat = tf.compat.v1.transpose(x_hat, perm=[0, 2, 3, 1])
        if distance=='l2-lpips':
            x_hat = tf.compat.v1.image.grayscale_to_rgb(x_hat)        

        ### load model
        vars = [v for v in tf.compat.v1.global_variables() if 'latent_z' not in v.name]
        saver = tf.compat.v1.train.Saver(vars)
        sess.run(tf.compat.v1.variables_initializer(vars))
        if_load = load_model_from_checkpoint(load_dir, saver, sess)
        assert if_load is True

        ### loss
        if distance == 'l2':
            print('use distance: l2')
            loss_l2 = tf.compat.v1.reduce_mean(tf.compat.v1.square(x_hat - x), axis=[1, 2, 3])
            vec_loss = loss_l2
            vec_losses = {'l2': loss_l2}
        elif distance == 'l2-lpips':
            print('use distance: lpips + l2')
            loss_l2 = tf.compat.v1.reduce_mean(tf.compat.v1.square(x_hat - x), axis=[1, 2, 3])
            loss_lpips = lpips_tf.lpips(x_hat, x, normalize=False, model='net-lin', net='vgg', version='0.1')
            vec_losses = {'l2': loss_l2, 'lpips': loss_lpips}
            vec_loss = loss_l2 + LAMBDA2 * loss_lpips
        else:
            raise NotImplementedError

        ## regularizer
        norm = tf.compat.v1.reduce_sum(tf.compat.v1.square(z), axis=1)
        norm_penalty = (norm - Z_DIM) ** 2

        if if_norm_reg:
            loss = tf.compat.v1.reduce_mean(vec_loss) + LAMBDA3 * tf.compat.v1.reduce_mean(norm_penalty)
            vec_losses['norm'] = norm_penalty
        else:
            loss = tf.compat.v1.reduce_mean(vec_loss)

        ### set up optimizer
        opt = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[z],
                                                        method='L-BFGS-B',
                                                        options={'maxfun': maxfunc})

        ### run the optimization on query images
        query_loss, query_z, query_xhat = optimize_z(sess, z, x, x_hat,
                                                        init_val_ph, init_val['pos'],
                                                        pos_query_imgs,
                                                        check_folder(os.path.join(save_dir, 'pos_results')),
                                                        opt, vec_loss, vec_losses, BATCH_SIZE)
        save_files(save_dir, ['pos_loss'], [query_loss])

        query_loss, query_z, query_xhat = optimize_z(sess, z, x, x_hat,
                                                        init_val_ph, init_val['neg'],
                                                        neg_query_imgs,
                                                        check_folder(os.path.join(save_dir, 'neg_results')),
                                                        opt, vec_loss, vec_losses, BATCH_SIZE)
        save_files(save_dir, ['neg_loss'], [query_loss])

    save_dir = os.path.join(load_dir,'mia_results/wb')
