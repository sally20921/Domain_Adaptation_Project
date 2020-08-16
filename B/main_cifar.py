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
  if  args['datasets'] == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
  else:
    dataloader =  datasets.CIFAR100
    num_classes = 100

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
 
  trainset = dataloader(root=args['root'], train=True, download=True, transform = transfrom_train)
  trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

  testset =  dataloader(root=args['root'], train=False, download=False, transform=transform_test)
  testloader = data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers']

  ####################creating  model###############################3
