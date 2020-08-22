import os 
import numpy as np
import torch
import torchvision

from dataloader.data_transfrom import DataTransform

def load_data(args):
    print('Loading cifar10 data')
    train_dataset = torchvision.datasets.CIFAR10(
            args.data_path, 
            download=True, 
            transform=DataTransform(size=args.image_size)
    )

    train_iter =  torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.num_workers
    )

    return {'train': train_iter}

def get_iterator(args):
    iters = load_data(args)
    print('Data Loading Done')
    return iters



