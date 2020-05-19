import os
import glob
import torch
import h5py
import matplotlib.pyplot as plt
import os.path

import numpy as np
import pandas as pd
import random
import pickle
from datetime import datetime
from src.dataloader import CTWLabelledDataset

def read_raw_data_file(data_file, unlabelled = False):
    f = h5py.File(data_file, 'r')
    H_Re = f['H_Re'][:] #shape (sample size, 56, 924, 5)
    H_Im = f['H_Im'][:] #shape (sample size, 56, 924, 5) 
    SNR = f['SNR'][:] #shape (sample size, 56, 5)

    sample_size = len(H_Re)
    chunk = []

    if not unlabelled:
        Pos = f['Pos'][:] #shape(sample size, 3)
        for i in range(sample_size):
            chunk.append([H_Re[i], H_Im[i], SNR[i], Pos[i]])
    else:
        for i in range(sample_size):
            chunk.append([H_Re[i], H_Im[i], SNR[i]])
    f.close()
    
    return chunk
    
def process_and_save(name_, data_,  unlabelled = False):

    H_Re = [i[0] for i in data_]
    H_Im = [i[1] for i in data_]
    SNR = [i[2] for i in data_]
    if not unlabelled:
        Pos = [i[3] for i in data_]
        obj_ = {'H_Re': H_Re, 'H_Im': H_Im, 'SNR': SNR, 'Pos': Pos}
    else:
        obj_ = {'H_Re': H_Re, 'H_Im': H_Im, 'SNR': SNR}
    
    with open('../data/{}.pkl'.format(name_), 'wb') as f:
        pickle.dump(obj_, f)

    print("[SAVED] DATA: {}, Num of data: {}".format(name_, str(len(data_))))
    
def split_data(all_=False):

    random.seed(77)
    if (all_):
        labelled_data_file_all_list = glob.glob("../data/CTW2020_labelled_data/*.hdf5")

        total_num = 0
        labelled_data_all = []

        for data_file in labelled_data_file_all_list:
            chunk = read_raw_data_file(data_file, False)
            labelled_data_all += chunk
    else:
        one_sample_file = "../data/CTW2020_labelled_data/file_1.hdf5"
        labelled_data_all = read_raw_data_file(one_sample_file, False)


    total_num = len(labelled_data_all)
    print("TOTAL NUM OF LABELLED DATA: ", total_num, \
          "({})".format(len(labelled_data_all)))

    train_num = int(total_num*0.8)
    val_num = int((total_num - train_num)/2)

    shuffled_labelled_data_all = random.sample(labelled_data_all, len(labelled_data_all))

    labelled_data_train = shuffled_labelled_data_all[:train_num]
    labelled_data_valid =shuffled_labelled_data_all[train_num:train_num+val_num]
    labelled_data_test = shuffled_labelled_data_all[train_num+val_num:total_num+1]

    print("<< SHUFFLE AND DIVIDE INTO TRAIN, VAL, TEST >>")
    print("[TOTAL: {}] TRAIN: {}, VAL: {}, TEST: {}"\
          .format(len(labelled_data_train) + len(labelled_data_valid) + len(labelled_data_test)\
                  , len(labelled_data_train), len(labelled_data_valid), len(labelled_data_test)))
                
    data_col = [\
        ('labelled_train', labelled_data_train),
        ('labelled_valid', labelled_data_valid),
        ('labelled_test', labelled_data_test),
    ]
    
    for data_name, splitted_data in data_col:
        process_and_save(data_name, splitted_data, False)

        # by end of this, each pickle file contains dictionary of
        # obj_ = {'H_Re': H_Re, 'H_Im': H_Im, 'SNR': SNR, 'Pos': Pos}
        
    print("[DONE] Splited and Saved into 80%, 10%, 10%")

def read_pickle_file(picklepath):
    with open(picklepath, 'rb') as f:
        data = pickle.load(f)

    H_Re = data['H_Re']
    H_Im = data['H_Im']
    SNR = data['SNR']
    try:
        Pos = data['Pos']
    except:
        Pos = None

    print("Reading from: {}, Num of data: {}".format(picklepath, len(H_Re)))

    return H_Re, H_Im, SNR, Pos

# def split_train_file():

def prepare_unlabelled_data(k = 10):
    random.seed(77)

    unlabelled_data_file_all_list = glob.glob("../data/CTW2020_unlabelled_data/*.hdf5")
    random.shuffle(unlabelled_data_file_all_list)

    print("Preparing unlabelled data with {} sample files".format(k))
    unlabelled_data_list = unlabelled_data_file_all_list[:k]
    unlabelled_data_all = []
    for data_file in unlabelled_data_list:
        chunk = read_raw_data_file(data_file, unlabelled=True)
        unlabelled_data_all += chunk

    total_num = len(unlabelled_data_all)
    print("TOTAL NUM OF UNLABELLED DATA: ", total_num, \
          "({})".format(len(unlabelled_data_all)))

    process_and_save('unlabelled', unlabelled_data_all, True)

def combine_unlabel_and_labelled_data(pred_path, unlabelled_path):
    pred_ = np.load(pred_path)
    unlabelled_data = torch.load(unlabelled_path)

    unlabelled_data.Pos = pred_
    labelled_unlabelled_dataset = unlabelled_data

    return labelled_unlabelled_dataset

if __name__ == '__main__':

# ==  RUN1
# ==  Splitting all labelled data into train, valid, test set
    # select_all_data = True
    # split_data(select_all_data)

#==  Preparing Un]labelled data
    # prepare_unlabelled_data(k=10)


#== label unlabelled data:

    pred_path = '/home/etsuko/comp5212_term_project/save/pred/unlabelled_pred.npy'
    unlabelled_path = '/home/yejin/comp5212_term_project/data/all_data/unlabelled_dataset.pth'

    labelled_unlabelled_dataset = combine_unlabel_and_labelled_data(pred_path, unlabelled_path)
    torch.save(labelled_unlabelled_dataset, '/home/yejin/comp5212_term_project/data/all_data/labelled_unlabelled_dataLoader.pth')

