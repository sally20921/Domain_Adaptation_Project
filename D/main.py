import argparse
import numpy as np
import os

import pandas as pd
import torch
import torchvision
import torch.optim as optim 
from torch.utils.data import DataLoader
from tqdm import tqdm

import .data_augmentation import CIFAR10Pair, train_transform, test_transform

from torch.utils.tensorboard import SummaryWriter

from .simclr import SimCLR
from .nt_xent import NT_Xent

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    global best_acc
    start_epoch = 0

    if not os.path.isdir(args['checkpoint']):
        os.makedirs(arg['checkpoint'])

    tfboard_dir = os.path.join(args['checkpoint'], 'tfboard')
    if not os.path.isdir(tfboard_dir):
        os.makedirs(tfboard_dir)
    writer = SummaryWriter(tfboard_dir)

    # data loading
    train_data = CIFAR10Pair(root=args['root'], train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], pin_memory=True, drop_last=True)

    test_data = CIFAR10Pair(root=args['root'], train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True)

    # model creation
    model = SimCLR(args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion =  NT_Xent(args)
    # train 
    for epoch in range(1, args['epochs']+1):
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, writer)
        test_loss, test_acc = test(model, test_loader, optimizer)


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for (x_i, x_j, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)
        
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        writer.add_scalar("Loss/train_epoch", loss.item())
        loss_epoch += loss.item()

    return loss_epoch


