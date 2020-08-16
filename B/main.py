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
  if args['datasets'] == 'imagenet':
    traindir = os.path.join(args['root'], 'train')
    valdir = os.path.join(args['root'], 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456,  0.406],
                                     std=[0.229,0.224,0.225])
    train_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        normalize,
      ])),
      batch_size=args['batch_size'], shuffle=True, 
      num_workers=args['num_workers'], pin_memory=True)

     val_loader = torch.utils.data.DataLoader(
       datasets.ImageFolder(valdir, transforms.Compose([
         transforms.Scale(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize,
       ])),
       batch_size=args['batch_size'], shuffle=False,
       num_workers=args['num_workers'], pin_memory=True)


