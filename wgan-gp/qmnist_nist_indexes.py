# This script is used to get the correspondance between qmnist and nist indexes

import os
import pickle
import argparse

import numpy as np
import pandas as pd
from torchvision.datasets import QMNIST

def get_digit_indexes(byIMG_df, qmnist_labels):
    nist_indexes = byIMG_df.iloc[:,3].to_numpy()
    nist_paths = byIMG_df.iloc[:,4].to_list()
    qmnist_indexes = []
    for path in nist_paths:
        if isinstance(path,list):
            path = path[0]
        image_id = path.split('/')[-1]
        image_id_list = image_id.replace('.png','').split('_')
        writer_id = int(image_id_list[0][1:])
        writer_digit_index = int(image_id_list[-1])
        qmnist_index = np.where((qmnist_labels[:,2]==writer_id) & (qmnist_labels[:,3]==writer_digit_index))[0][0]
        qmnist_indexes.append(qmnist_index)
    return np.array(qmnist_indexes), nist_indexes

def get_byIMG_df(nist_datapath):
    byIMG_df = pd.DataFrame()
    for hsf in [0,1,2,3,4,6,7]:
        with open(os.path.join(nist_datapath,'/HSF_'+str(hsf)+'_byIMG.pkl'),'rb') as f:
            byIMG = pickle.load(f)
            if hsf==0:
                index_offset = 0
            else:
                index_offset = index_offset + byIMG_df[byIMG_df['hsf'] == previous_hsf].shape[0]
            byIMG['hsf'] = hsf
            byIMG[3] = byIMG[3]+index_offset
            byIMG_df = pd.concat((byIMG_df, byIMG), ignore_index=True)
            previous_hsf=hsf
    return byIMG_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get index correspondance between QMNIST and NIST. It takes around 25 mintues')
    parser.add_argument('--nist_datapath', help='Path for nist data', required=True)
    parser.add_argument('--save_path', help='Path to save the pickle with the index correspondence', default=None)
    args = parser.parse_args()

    if args.save_path is None:
        save_path = args.nist_datapath
    else:
        save_path = args.save_path

    qmnist_data = QMNIST('qmnist', download=True, what='nist')
    qmnist_labels = qmnist_data.targets.numpy()

    byIMG_df = get_byIMG_df(args.nist_datapath)

    qmnist_indexes, nist_indexes = get_digit_indexes(byIMG_df, qmnist_labels)

    qmnist_nist_indexes = dict()
    qmnist_nist_indexes['nist_indexes'] = nist_indexes
    qmnist_nist_indexes['qmnist_indexes'] = qmnist_indexes
    with open(os.path.join(save_path,'qmnist_nist_indexes.pickle'), 'wb') as f:
        pickle.dump(qmnist_nist_indexes, f)