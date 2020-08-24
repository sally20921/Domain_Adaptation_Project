from collections import defaultdict
import torch
from utils import *

import os 
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

modes = ['train', 'val', 'test']

class ImageData(Dataset):
    def __init__(self, args, mode):
        if mode not in modes:
            raise ValueError("mode should be %s." % (' or '.join(modes)))

        self.args = args 
        self.mode = mode 

    def __len__(self):
    
    def __getitem__(self, idx):
        data =  {
            'data':
            'target':
        }
        return data

def load_data(args):
    print("Loading data")
    train_dataset  = ImageData(args, mode='train')
    valid_dataset = ImageData(args, mode='val')
    test_dataset = ImageData(args, mode='test')

    train_iter = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers
    )

    val_iter = DataLoader(
            valid_dataset, 
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers
    )

    test_iter = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle= args.shuffle, 
            num_workers=args.num_workers
    )

    return  {'train': train_iter, 'val': val_iter, 'test': test_iter}

def get_iterator(args):
    iters = load_data(args)
    print("data loading done")

    return iters 
