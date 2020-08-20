from collections import defaultdict
import torch
from utils import *

import os 
import re
from tqdm import tqdm
import numpy as np
import random

import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

modes = ['train', 'val', 'test']


def load_data(args):
    print("Loading data")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader  = datasets.CIFAR100
    num_classes = 100

    train_dataset = dataloader(root=args.data_path, train=True, download=True, transform=  transform_train)
    test_dataset = dataloader(root=args.data_path, train=False, download=False, transform=transform_test)
    
    train_iter = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size[0],
            shuffle=args.shuffle[0],
            num_workers=args.num_workers
    )

    test_iter = data.DataLoader(
            test_dataset, 
            batch_size=args.batch_size[1],
            shuffle= args.shuffle[1], 
            num_workers=args.num_workers
    )

    return  {'train': train_iter, 'test': test_iter}

def get_iterator(args):
    iters = load_data(args)
    print("data loading done")

    return iters 
