from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
# from src.data import read_pickle_file
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

# TRAIN_PICKLE = './data/labelled_train.pkl'
# VALID_PICKLE = './data/labelled_valid.pkl'
# TEST_PICKLE = './data/labelled_test.pkl'

TRAIN_PICKLE = './data/all_data/labelled_train.pkl'
VALID_PICKLE = './data/all_data/labelled_valid.pkl'
TEST_PICKLE = './data/all_data/labelled_test.pkl'
UNLABELLED_PICKLE = './data/all_data/unlabelled.pkl'

# TRAIN_DATALOADER = './data/all_data/train_dataLoader.pth'
# VALID_DATALOADER = './data/all_data/valid_dataLoader.pth'
# TEST_DATALOADER = './data/all_data/test_dataLoader.pth'
# UNLABELLED_DATALOADER = './data/all_data/unlabelled_dataLoader.pth'

# === In case you are using eez115
TRAIN_DATALOADER = '/home/yejin/comp5212_term_project/data/all_data/train_dataLoader.pth'
VALID_DATALOADER = '/home/yejin/comp5212_term_project/data/all_data/valid_dataLoader.pth'
TEST_DATALOADER = '/home/yejin/comp5212_term_project/data/all_data/test_dataLoader.pth'
UNLABELLED_DATALOADER = '/home/yejin/comp5212_term_project/data/all_data/unlabelled_dataLoader.pth'


class CTWLabelledDataset(Dataset):
    def __init__(self, H_Re, H_Im, SNR, Pos):
        self.H_Re = torch.tensor(H_Re)
        self.H_Im = torch.tensor(H_Im)
        self.SNR = torch.tensor(SNR)
        if (Pos is not None):
            self.Pos = torch.tensor(Pos)
        else:
            self.Pos = None

    def __len__(self):
        return len(self.H_Re)

    def __getitem__(self, idx):
        if (self.Pos is not None):
            return self.H_Re[idx], self.H_Im[idx], self.SNR[idx], self.Pos[idx]
        else:
            return self.H_Re[idx], self.H_Im[idx], self.SNR[idx]

def get_dataset(type ='train'):
    print(type)
    if type == 'train':
        H_Re, H_Im, SNR, Pos = read_pickle_file(TRAIN_PICKLE)
    elif type == 'valid':
        H_Re, H_Im, SNR, Pos= read_pickle_file(VALID_PICKLE)
    elif type == 'test':
        H_Re, H_Im, SNR, Pos= read_pickle_file(TEST_PICKLE)
    elif type == 'unlabelled':
        H_Re, H_Im, SNR, Pos = read_pickle_file(UNLABELLED_PICKLE)
    else:
        raise ValueError('Invalid type={type}')

    return CTWLabelledDataset(H_Re, H_Im, SNR, Pos)

def get_dataLoader(dataLoader_saved=True, bs=32, use_concat=False):

    if dataLoader_saved == False:
        print("Makeing dataset and  dataloader and Using default batchsize 32")
        train_dataLoader, valid_dataLoader, test_dataLoader = get_and_save_dataLoader(bs)

    else:
        valid_dataLoader = torch.load(VALID_DATALOADER)
        test_dataLoader = torch.load(TEST_DATALOADER)

        if use_concat:
            print("Loading saved dataloader with concat train")

            c = torch.load('/home/yejin/comp5212_term_project/data/all_data/concat_dataset.pth')
            train_dataLoader = DataLoader(c, batch_size=32, shuffle=True)
        else:
            print("Loading saved dataloader with only labelled train")
            train_dataLoader = torch.load(TRAIN_DATALOADER)

    print("Save!")
    dataloader_col = {
        'train':train_dataLoader,
        'valid':valid_dataLoader,
        'test':test_dataLoader
    }

    return dataloader_col

def get_unlabelled():
    return torch.load(UNLABELLED_DATALOADER)

def get_and_save_dataLoader(bs=32):

    train_dataLoader = DataLoader(get_dataset('train'), batch_size=bs, shuffle=True)
    torch.save(train_dataLoader, 'train_dataLoader.pth')
    now = datetime.now()
    print("Saved: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))

    valid_dataLoader =  DataLoader(get_dataset('valid'), batch_size=bs, shuffle=False)
    torch.save(valid_dataLoader, 'valid_dataLoader.pth')
    now = datetime.now()
    print("Saved: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))

    test_dataLoader = DataLoader(get_dataset('test'), batch_size=bs, shuffle=False)
    torch.save(test_dataLoader, 'test_dataLoader.pth')
    now = datetime.now()
    print("Saved: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))

    return train_dataLoader, valid_dataLoader, test_dataLoader

def make_unlabelled_dataLoader(bs=32):
    now = datetime.now()
    print("Staet: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))

    unlabelled_dataset = get_dataset('unlabelled')
    torch.save(unlabelled_dataset, 'unlabelled_dataset.pth')
    now = datetime.now()
    print("Saved 1: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))
    # unlabelled_dataset = torch.load('unlabelled_dataset.pth')

    unlabelled_dataLoader = DataLoader(unlabelled_dataset, batch_size=bs, shuffle=True)
    torch.save(unlabelled_dataLoader, 'unlabelled_dataLoader.pth')

    now = datetime.now()
    print("Saved 2: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))

if __name__ == '__main__':
    # make_unlabelled_dataLoader(bs=32)
    for type in ['train', 'valid', 'test']:
        now = datetime.now()
        print("Start: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))
        dataset = get_dataset(type)
        now = datetime.now()
        print("Saving{}: {}".format(type, now.strftime("%d/%m/%Y %H:%M:%S")))

        torch.save(dataset, './data/all_data/{}_dataset.pth'.format(type))
