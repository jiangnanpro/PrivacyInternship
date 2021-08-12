import os
import sys
import pickle
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))
sys.path.append(os.path.join(os.path.dirname(file_path),'attacks/tools/lpips_tensorflow'))

import numpy as np
from torchvision.datasets import QMNIST
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

from attacks.fbb import find_knn, find_pred_z
from tflib.nist import load_nist_images
from tflib.utils import save_files

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

n_images = nist_indexes.shape[0]

# Randomly selecting the images for each partition
rng = np.random.RandomState(2021)
random_seq = rng.choice(n_images,size=n_images, replace=False)


K = 5
BATCH_SIZE = 10

training_set_sizes = [128, 256, 412, 1024, 2048, 4096, 8192, 16384]
for n in training_set_sizes:
    load_dir = os.path.join(os.path.dirname(file_path),'models/wgan-gp_nist{}'.format(n))
    save_dir = os.path.join(load_dir,'mia_results/fbb')

    generate = np.load(os.path.join(load_dir, 'generated.npz'))
    gen_imgs = generate['images']
    gen_z = generate['noise']
    gen_feature = np.reshape(gen_imgs, [len(gen_imgs), -1])
    gen_feature = 2. * gen_feature - 1.

    defender_partition_size = n
    reserve_partition_size = n_images-defender_partition_size

    defender_partition_indexes = random_seq[0:defender_partition_size]
    reserve_partition_indexes = random_seq[defender_partition_size:]

    x_defender_nist = nist_images[nist_indexes[defender_partition_indexes]]
    x_defender_qmnist = qmnist_images[qmnist_indexes[defender_partition_indexes]]

    x_reserve_nist = nist_images[nist_indexes[reserve_partition_indexes]]
    x_reserve_qmnist = qmnist_images[qmnist_indexes[reserve_partition_indexes]]

    ### load data
    ### FULL BLACK-BOX ATTACK

    ### load data
    DATA_NUM = min(n,1000)
    pos_query_imgs = x_defender_nist[:DATA_NUM]
    neg_query_imgs = x_reserve_nist[:DATA_NUM]

    ### nearest neighbor search
    nn_obj = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn_obj.fit(gen_feature)

    ### positive query
    pos_loss, pos_idx = find_knn(nn_obj, pos_query_imgs)
    pos_z = find_pred_z(gen_z, pos_idx)
    save_files(save_dir, ['pos_loss', 'pos_idx', 'pos_z'], [pos_loss, pos_idx, pos_z])

    ### negative query
    neg_loss, neg_idx = find_knn(nn_obj, neg_query_imgs)
    neg_z = find_pred_z(gen_z, neg_idx)
    save_files(save_dir, ['neg_loss', 'neg_idx', 'neg_z'], [neg_loss, neg_idx, neg_z])
