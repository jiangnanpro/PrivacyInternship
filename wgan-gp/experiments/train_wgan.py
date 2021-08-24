import os
import sys
import pickle
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))

import numpy as np
from torchvision.datasets import QMNIST
import tensorflow as tf
if tf.__version__[0]=='2':
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
tf.compat.v1.set_random_seed(2021)

from tflib.gan import train
from tflib.nist import load_nist_images
from tflib.utils import check_folder, shuffle_in_unison

"""### Get qmnist-nist index correspondence"""

nist_datapath = '/home/rafael07/privacy_benchmark/data/nist'
qmnist_datapath = '/home/rafael07/privacy_benchmark/data/qmnist'
nist_qmnist_indexes_datapath = os.path.join(file_path,'qmnist_nist_indexes.pickle')

with open(nist_qmnist_indexes_datapath, 'rb') as f:
    qmnist_nist_dict = pickle.load(f)
qmnist_indexes = qmnist_nist_dict['qmnist_indexes']
nist_indexes = qmnist_nist_dict['nist_indexes']

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

shuffle_in_unison(qmnist_indexes, nist_indexes)

training_set_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

for n in training_set_sizes:

    x_defender_nist = nist_images[nist_indexes[:n]]
    y_defender_nist = nist_labels[nist_indexes[:n]].squeeze()
    x_defender_qmnist = qmnist_images[qmnist_indexes[:n]]
    y_defender_qmnist = qmnist_labels[qmnist_indexes[:n]]

    INPUT_HEIGHT = 28
    INPUT_WIDTH = 28
    OUTPUT_DIM = INPUT_HEIGHT*INPUT_WIDTH

    BATCH_SIZE = 32
    MODE = 'wgan-gp'
    CRITIC_ITERS = 5
    LAMBDA = 10
    DIM = 64
    ITERS = 100000
    TRAIN_WITH_DP = False
    L2_NORM_CLIP = None
    NOISE_MULTIPLIER = None
    MODEL_PATH_NIST = os.path.join(os.path.dirname(file_path),'models/wgan-gp_nist_{}'.format(n))
    MODEL_PATH_QMNIST = os.path.join(os.path.dirname(file_path),'models/wgan-gp_qmnist_{}'.format(n))

    check_folder(MODEL_PATH_NIST)
    check_folder(MODEL_PATH_QMNIST)
    
    train(x_defender_qmnist/255, y_defender_qmnist, INPUT_WIDTH, INPUT_HEIGHT, MODEL_PATH_QMNIST, 
        BATCH_SIZE, DIM, MODE, LAMBDA, CRITIC_ITERS, ITERS, TRAIN_WITH_DP)

    train(x_defender_nist/255, y_defender_nist, INPUT_WIDTH, INPUT_HEIGHT, 
        MODEL_PATH_NIST, BATCH_SIZE, DIM, MODE, LAMBDA, CRITIC_ITERS, ITERS, TRAIN_WITH_DP)