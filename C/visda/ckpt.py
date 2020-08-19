import os

import torch, pickle
from torch import nn
import torch.nn.functional as F

from dataloader.load_dataset import get_iterator
from model import get_model 
from utils import get_dirname_from_args

def get_ckpt_path(args, epoch, loss):
    ckpt_name = get_dirname_from_args(args)
    ckpt_path = args.ckpt_path / ckpt_name
    args.ckpt_path.mkdir(exist_ok=True)
    ckpt_path.mkdir(exist_ok=True)
    loss = '{:4f}'.format(loss)
    ckpt_path =  ckpt_path / 'loss_{}_epoch_{}.pickle'.format(loss,epoch)
    return ckpt_path


