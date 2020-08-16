import os
import torch
import torchvision
import argparse
import yaml
from torch.utils import data

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#@ex.automain
def main():
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
  
  ############Data Loading#########################################
  if args['datasets'] == 'cifar10':
    dataloader =  datasets.CIFAR10
    num_classes = 10

  elif args['datasets'] == 'cifar100':
    dataloader =  datasets.CIFAR100
    num_classes = 100

  else:
    pass

  

